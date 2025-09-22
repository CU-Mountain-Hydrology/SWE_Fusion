"""
Automatically generates report maps in ArcPy using post-processed rasters.

**Usage:**
    python generate_maps.py report_type date [--figs REGEX] [--preview]

**Arguments:**
    - ``report_type``: Type of report to generate (e.g., WW)
    - ``date``: Date of report in YYYYMMDD format

**Options:**
    - ``--figs=REGEX``: Regex for figures to generate (default: all).
      Example: ``--figs=1a`` or ``--figs=1*,2a,3``
    - ``--preview``, ``-p``: Open the generated JPG maps upon completion
    - ``--verbose``, ``-v``: Enable verbose output messages
    - ``--prompt_user``, ``-u``: Prompt user before overwriting files or automatically selecting layer files
"""

#########################   These values should not need to be changed between runs, but may change depending on your
#         CONFIG        #   operating system, filepaths, and preferred output location.
#########################
# Filepath configs
template_aprx = "U:\EricG\MapTemplate\MapTemplate.aprx" # Project containing template for each figure
product_source_dir = r"U:\EricG\testing_Directory"      # Parent directory of the YYYYMMDD_RT_Report folders
output_parent_dir = "../output/"                        # Directory the figures will be exported to

# Figure configs
layer_formats = ["tif",]                                # List of all layer file formats used in the ww_fig_data dict
table_formats = ["csv","dbf",]                          # List of all standalone table file formats
ww_fig_data = {                                         # Dictionary of metadata for all figures in the WestWide reports
    # The top level contains each figure id as it appears in the report, followed by an id for each map-frame within that
    # figure. For each map, the files that need updated are specified with an id such as "p8", and its format as well as
    # source directory (*UseThis or *UseAvg) are specified.
    "1a": {
        "maps": {
            "1a": [
                {"layer": "p8", "format": "tif", "dir": "*UseThis"}
            ]
        }
    },
    "1b": {
        "maps": {
            "1b": [
                {"layer": "anomRegion_table", "format": "csv", "dir": "*UseAvg"}
            ]
        }
    },
    "2a": {
        "maps": {
            "2a" : [
                {"layer": "anom0_200_msk", "format": "tif", "dir": "*UseAvg"}
            ]
        }
    },
    "2b": {
        "maps": {
            "2b": [
                {"layer": "p11", "format": "tif", "dir": "*UseAvg"},
                {"layer": "huc6_anom_table_save", "format": "dbf", "dir": "*UseAvg"},
            ]
        }
    },
    "3": {
        "maps": {
            "3a": [
                {"layer": "p8", "format": "tif", "dir": "*UseThis"}
            ],
            "3b": [
                {"layer": "anom0_200_msk", "format": "tif", "dir": "*UseAvg"}
            ],
            "3c": [
                {"layer": "p9", "format": "tif", "dir": "*UseAvg"},
                {"layer": "anom_table_save", "format": "dbf", "dir": "*UseAvg"}, # TODO: NOT huc6_anom_table_save, will find both when parsing
            ],
            "3d": []
        }
    },
    "4": {
        "maps": {
            "4a": [
                {"layer": "p8", "format": "tif", "dir": "*UseThis"}, # TODO: In the template, this is not from UseThis or UseAvg?
                # TODO: 3 more standalone tables in the template: zonal_stats and watersheds_elev_utm_stat
            ],
            "4b": [
                {"layer": "anom0_200_msk", "format": "tif", "dir": "*UseAvg"},
            ],
            "4c": [
                {"layer": "p9", "format": "tif", "dir": "*UseAvg"},
                {"layer": "anom_table_save", "format": "dbf", "dir": "*UseAvg"},
            ],
            "4d": []
        }
    },
    "5": {
        "maps": {
            "5a": [
                {"layer": "p8", "format": "tif", "dir": "*UseThis"}
            ],
            "5b": [
                {"layer": "anom0_200_msk", "format": "tif", "dir": "*UseAvg"}
            ],
            "5c": [
                {"layer": "p9", "format": "tif", "dir": "*UseAvg"},
                {"layer": "anom_table_save", "format": "dbf", "dir": "*UseAvg"},
            ],
            "5d": []
        }
    },
    "6": {
        "maps": {
            "6a": [
                {"layer": "p8", "format": "tif", "dir": "*UseThis"}
            ],
            "6b": [
                {"layer": "anom0_200_msk", "format": "tif", "dir": "*UseAvg"}
            ],
            "6c": [
                {"layer": "p9", "format": "tif", "dir": "*UseAvg"},
                {"layer": "anom_table_save", "format": "dbf", "dir": "*UseAvg"},
            ],
            "6d": []
        }
    },
}

#########################
#       END CONFIG      #
#########################

from zero_to_no_data import *
import argparse
import re # Regular Expression
import os # Operating System
import glob # OS Pattern Searching
import tempfile
import shutil
import arcpy

def interpret_figs(figs: str, report_type: str) -> list[str]:
    """
    Interprets the --figs regex flag

    :param figs: String value of the --figs flag as generated by parser.parse_args()
    :param report_type: Type of report to interpret figures for. e.g., WW
    :return: List of interpreted figure names to generate maps for
    :rtype: list[str]
    """
    # Determine list of figures based on report type
    match report_type:
        case 'WW':
            all_figs = set(ww_fig_data.keys())
        case _:
            raise Exception(f"Unrecognized report type: {report_type}")

    # Parse the argument passed into --figs
    patterns = figs.split(",")
    fig_list = set()

    for pattern in patterns:
        # Shortcut search when all figs are specified
        if pattern in ["all","."]:
            return sorted(all_figs)

        # Modify regular expression syntax to better support * wildcard
        regex_pattern = "^" + re.escape(pattern).replace("\\*", ".*") + "$"
        regex = re.compile(regex_pattern)

        # Match the pattern against all possible figure names
        pattern_found = False
        for fig in all_figs-fig_list:
            if regex.match(fig):
                fig_list.add(fig)
                pattern_found = True
        if not pattern_found:
            # Pattern does not match any name in all_figs
            raise Exception(f"--figs pattern {pattern} not recognized!")

    return sorted(fig_list)


def find_layer_file(date: int, layer_info: dict, prompt_user = True, warn = True) -> str:
    """
    Finds the specific layer file to use

    :param date: Date of report in YYYYMMDD format
    :param layer_info: Dictionary containing the layer id, format, and directory
    :param prompt_user: Enable prompting the user when selecting between multiple options. Default: True
    :param warn: Enable warning messages. Default: True
    :return: Path to the layer file
    :rtype: str
    """

    # Extract layer metadata from layer_info
    layer_id = layer_info["layer"]
    file_type = layer_info["format"]
    dir_pattern = layer_info["dir"]

    # Find RT_Report directory for this date
    # TODO: may want to make this the results dir not the RT_Report dir, need to figure out what data is duplicated where
    # TODO: ^^ good practice to copy files out of results dir to avoid corruption problems. Automate this process
    rt_report_dir = os.path.join(product_source_dir, str(date) + "_RT_Report")

    # Find the directory containing the layer products to be used e.g. "...UseThis"
    try:
        layer_dir = glob.glob(os.path.join(rt_report_dir, dir_pattern))[0]
    except IndexError:
        raise FileNotFoundError(f"No directory matching pattern '{dir_pattern}' found in '{rt_report_dir}'! "
                                f"Confirm config values are set correctly.")

    # Find the layer products that contains the layer_id e.g. "p8"
    layer_files = glob.glob(os.path.join(layer_dir, f"*{layer_id}*.{file_type}"))
    if not layer_files:
        raise FileNotFoundError(f"No file matching pattern '{layer_id}' found in '{layer_dir}'!")

    layer_file = layer_files[0]
    if len(layer_files) > 1:
        # If one of the multiple layer files ends with "nulled", use it
        for file in layer_files:
            if file_type == "tif" and file.endswith("nulled.tif"):
                # TODO: option to disable selecting "...nulled.tif" by default
                layer_file = file
                if warn: print(f"find_layer_file warning: Multiple files matching pattern '{layer_id}' found in '{layer_dir}'. Using {layer_file}")
                break
        else: # If none of the layer files end with "nulled"
            if prompt_user:
                # Ask the user to select which file to use
                print(f"Multiple files matching pattern '{layer_id}' found in '{layer_dir}'.")
                for i, file in enumerate(layer_files):
                    print(f"\t{i+1}. {file}")
                while True:
                    print(f"Enter a number from 1 to {len(layer_files)}:", end=" ")
                    result = input()
                    try:
                        result = int(result)
                        if result in range(1,len(layer_files)+1): break
                    except ValueError: pass
                # TODO: option to remember this choice in the future e.g. always use the p*custom_name.tif pattern if multiple exist and no "nulled"
                layer_file = layer_files[int(result)-1]
                print(f"Using {layer_file}")
            elif warn:
                # Use the first layer file found
                print(f"find_layer_file warning: Multiple files matching pattern '{layer_id}' found in '{layer_dir}'. Using {layer_files[0]}")

    return layer_file


def main():
    # Parse input arguments and flags, see top of file for argument usage examples
    parser = argparse.ArgumentParser()
    parser.add_argument("report_type", type=str, help="Acceptable report types: WW")
    parser.add_argument("date", type=int, help="Date to process (YYYYMMDD)")
    parser.add_argument("--figs", default="all", type=str, help="Regex pattern(s) for figure names to generate")
    parser.add_argument("-p","--preview", action="store_true", help="Preview the generated maps")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output messages")
    parser.add_argument("-u", "--prompt_user", action="store_true", help="Prompt the user before overwriting or automatically selecting files")
    args = parser.parse_args()

    # Interpret --figs flag and return a list of figure names to generate
    fig_list = interpret_figs(args.figs, args.report_type)

    # Clone the template aprx to a temporary directory
    temp_dir = tempfile.mkdtemp()
    working_aprx = os.path.join(temp_dir, "working_aprx.aprx")
    shutil.copyfile(template_aprx, working_aprx)
    aprx = arcpy.mp.ArcGISProject(working_aprx) # Open the working aprx in ArcPy
    for fig_id in fig_list:
        fig_data = ww_fig_data.get(fig_id)
        if not fig_data:
            raise ValueError(f"No metadata found for figure '{fig_id}'")

        # Find the map(s) for this figure
        for map_id, layers in fig_data["maps"].items():
            _map = aprx.listMaps(f"*{map_id}*")[0]
            if not _map:
                raise ValueError(f"No map matching '{map_id}' found in '{working_aprx}'!")

            # Process each layer in this map
            for layer_info in layers:
                layer_id = layer_info["layer"]
                file_type = layer_info["format"]

                # Find and remove undefined placeholder layers/tables
                symbology = None
                if file_type in layer_formats:
                    undefined_layer = _map.listLayers(f"*{layer_id}*")[0]
                    symbology = undefined_layer.symbology
                    _map.removeLayer(undefined_layer)
                elif file_type in table_formats:
                    undefined_table = _map.listTables(f"*{layer_id}*")[0]
                    _map.removeTable(undefined_table)

                # Find the new layer source
                new_layer_path = find_layer_file(args.date, layer_info, prompt_user=args.prompt_user)

                # Special handling for rasters with zero-valued cells
                if new_layer_path.endswith(".tif") and not new_layer_path.endswith("_nulled.tif") and contains_zero_value_cells(new_layer_path):
                    nulled_path = new_layer_path.replace(".tif", "_nulled.tif")
                    zero_to_no_data(new_layer_path, nulled_path, prompt_user=args.prompt_user, verbose=args.verbose)
                    new_layer_path = nulled_path

                # Set the data source and update the symbology
                _map.addDataFromPath(new_layer_path)
                if file_type in layer_formats:
                    layer = _map.listLayers(f"*{layer_id}*")[0]
                    layer.symbology = symbology

        # Export the layout to JPEG
        layout = aprx.listLayouts(f"*{fig_id}*")[0]
        if not layout:
            raise ValueError(f"No layout matching '{fig_id}' found in '{working_aprx}'!")
        layout.name = f"{args.date}_{args.report_type}_Fig{fig_id}"

        output_dir = os.path.join(output_parent_dir, f"{args.date}_{args.report_type}_JPEGmaps")
        os.makedirs(output_dir, exist_ok=True)
        layout.exportToJPEG(os.path.join(output_dir, layout.name + ".jpg"))

    # TODO: automatically zip the JPEGmaps folder at the end?

    # Clean up
    del aprx

if __name__ == '__main__':
    main()