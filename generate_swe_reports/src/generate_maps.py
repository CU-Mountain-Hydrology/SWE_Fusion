"""
Automatically generates report maps in ArcPy using post-processed rasters.

**Usage:**
    python generate_maps.py report_type date [--figs REGEX] [--preview]

**Arguments:**
    - ``report_type``: Type of report to generate (e.g., WW)
    - ``date``: Date of report in YYYYMMDD format

**Options:**
    - ``--figs=REGEX``: Regex for figures to generate (default: all).
      Example: ``--figs=1a`` or ``--figs=1*,2a,3`` or ``--figs=none``
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
layer_formats = ["tif","shp"]                           # List of all layer file formats
table_formats = ["csv","dbf",]                          # List of all standalone table file formats
ww_fig_data = {                                         # Dictionary of metadata for all figures in the WestWide reports
    # The top level contains each figure id as it appears in the report, followed by an id for each map-frame within that
    # figure. For each map, the files that need updated are specified with an id such as "p8", and its format, source
    # directory (*UseThis or *UseAvg), and labelling type (None, Anno, or Layer) are specified. Anno means there is a
    # separate annotation text that needs updated while Layer means some layer is joined to this source and has labels on.
    "1a": {
        "maps": {
            "1a": [
                {"layer": "p8", "format": "tif", "dir": "*UseThis", "label": "None"}
            ]
        }
    },
    "1b": {
        "maps": {
            "1b": [
                {"layer": "anomRegion_table", "format": "csv", "dir": "*UseAvg", "label": "Anno"}
            ]
        }
    },
    "2a": {
        "maps": {
            "2a" : [
                {"layer": "anom0_200_msk", "format": "tif", "dir": "*UseAvg", "label": "None"}
            ]
        }
    },
    "2b": {
        "maps": {
            "2b": [
                {"layer": "p11", "format": "tif", "dir": "*UseAvg", "label": "None"},
                {"layer": "huc6_anom_table_save", "format": "dbf", "dir": "*UseAvg", "label": ["WW_HUC6_albn83","name"]},
            ]
        }
    },
    "3": {
        "maps": {
            "3a": [
                {"layer": "p8", "format": "tif", "dir": "*UseThis", "label": "None"}
            ],
            "3b": [
                {"layer": "anom0_200_msk", "format": "tif", "dir": "*UseAvg", "label": "None"}
            ],
            "3c": [
                {"layer": "p9", "format": "tif", "dir": "*UseAvg", "label": "None"},
                {"layer": "anom_table_save", "format": "dbf", "dir": "*UseAvg", "label": ["*_Basins_albn83", "SrtName"]}, # TODO: NOT huc6_anom_table_save, will find both when parsing
            ],
            "3d": []
        }
    },
    "4": {
        "maps": {
            "4a": [
                {"layer": "p8", "format": "tif", "dir": "*UseThis", "label": "None"}, # TODO: In the template, this is not from UseThis or UseAvg?
                # TODO: 3 more standalone tables in the template: zonal_stats and watersheds_elev_utm_stat
            ],
            "4b": [
                {"layer": "anom0_200_msk", "format": "tif", "dir": "*UseAvg", "label": "None"},
            ],
            "4c": [
                {"layer": "p9", "format": "tif", "dir": "*UseAvg", "label": "None"},
                {"layer": "anom_table_save", "format": "dbf", "dir": "*UseAvg", "label": ["*_Basins_albn83", "SrtName"]},
            ],
            "4d": []
        }
    },
    "5": {
        "maps": {
            "5a": [
                {"layer": "p8", "format": "tif", "dir": "*UseThis", "label": "None"}
            ],
            "5b": [
                {"layer": "anom0_200_msk", "format": "tif", "dir": "*UseAvg", "label": "None"}
            ],
            "5c": [
                {"layer": "p9", "format": "tif", "dir": "*UseAvg", "label": "None"},
                {"layer": "anom_table_save", "format": "dbf", "dir": "*UseAvg", "label": ["*_Basins_albn83", "SrtName"]},
            ],
            "5d": []
        }
    },
    "6": {
        "maps": {
            "6a": [
                {"layer": "p8", "format": "tif", "dir": "*UseThis", "label": "None"}
            ],
            "6b": [
                {"layer": "anom0_200_msk", "format": "tif", "dir": "*UseAvg", "label": "None"}
            ],
            "6c": [
                {"layer": "p9", "format": "tif", "dir": "*UseAvg", "label": "None"},
                {"layer": "anom_table_save", "format": "dbf", "dir": "*UseAvg", "label": ["*_Basins_albn83", "SrtName"]},
            ],
            "6d": []
        }
    },
}

snm_fig_data = {
    "0": {
        "maps": {
            "regions": [
                {"layer": "anomRegion_table", "format": "csv", "dir": "*UseAvg", "label": "Anno"}
            ]
        }
    },
    "1": {
        "maps": {
            "real_time_swe": [
                {"layer": "p8", "format": "tif", "dir": "*UseThis", "label": "None"}
            ],
            "anomaly": [
                {"layer": "anom0_200_msk", "format": "tif", "dir": "*UseAvg", "label": "None"}
            ],
            "watershed_map": [
                {"layer": "p9", "format": "tif", "dir": "*UseAvg", "label": "None"}
            ]
        }
    },
    "2": {
        "maps": {
            "ASO_SWE_diff": [
                # TODO: not sure how to automate updating ASO diff layers
            ]
        }
    },
    # Fig 3 is just fig 1 from the previous report, nothing to do here
    "4": {
        "maps": {
            "real_time_swe_fires": [
                {"layer": "p8", "format": "tif", "dir": "*UseThis", "label": "None"}
            ]
        }
    },
    "5": {
        "maps": {
            "TC_MODIS_image": [
                # TODO: not sure how to automate MODIS layer
            ]
        }
    },
    "6": {
        "maps": {
            "SNODAS_swe": [
                {"layer": "Cp_m_albn83_clp", "format": "tif", "dir": "SNODAS", "label": "None"}
            ],
            "SNODAS_diff": [
                {"layer": "SNODAS_Regress", "format": "tif", "dir": "*UseThis", "label": "None"}
            ],
            "SNODAS_swe_overlap": [
                {"layer": "both", "format": "tif", "dir": "SNODAS", "label": "None"}
            ]
        }
    },
    "7": {
        "maps": {
            "mean_swe": [
                {"layer": "mean_msk", "format": "tif", "dir": "*UseThis", "label": "None"},
                # TODO: add support for shp files
                {"layer": "Zero_sensors", "format": "shp", "dir": "", "label": "None"},
                {"layer": "sensors_SNM", "format": "shp", "dir": "", "label": "None"},
                {"layer": "Zero_CCR", "format": "shp", "dir": "", "label": "None"},
                {"layer": "CCR", "format": "shp", "dir": "", "label": "None"},
            ],
            "banded_elev": []
        }
    }
}

#########################
#       END CONFIG      #
#########################

from zero_to_no_data import *
from utils import crop_whitespace
import argparse
import re # Regular Expression
import os # Operating System
import glob # OS Pattern Searching
import tempfile
import shutil
import arcpy
from datetime import datetime

def interpret_figs(figs: str, report_type: str) -> list[str]:
    """
    Interprets the --figs regex flag

    :param figs: String value of the --figs flag as generated by parser.parse_args()
    :param report_type: Type of report to interpret figures for. e.g., WW
    :return: List of interpreted figure names to generate maps for
    :rtype: list[str]
    """
    # Determine list of figures based on report type
    all_figs = set()
    match report_type:
        case 'WW':
            all_figs = set(ww_fig_data.keys())
        case _:
            raise Exception(f"Unrecognized report type: {report_type}")

    # Parse the argument passed into --figs
    patterns = figs.split(",")
    fig_list = set()

    for pattern in patterns:
        pattern = pattern.lower()
        # Shortcut search when all figs are specified
        if pattern in ["all","."]:
            return sorted(all_figs)
        elif pattern in ["none",""]:
            return []

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
            if pattern == "0":
                raise Exception(f"--figs pattern '0' not recognized! Did you mean 'none'?")
            else:
                raise Exception(f"--figs pattern '{pattern}' not recognized!")

    return sorted(fig_list)


def find_layer_file(date: int, layer_info: dict, prompt_user = True, warn = True) -> str:
    """
    Finds the specific layer file to use

    :param date: Date of report in YYYYMMDD format
    :param layer_info: Dictionary containing the layer id, format, directory, and label type
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

def get_modis_date(date: int) -> int:
    # TODO: Go through all snapshot-date files in the directory and use ... most recent? idk how they are decided
    # Then format into YYYYMMDD
    return 20250326


def generate_maps(report_type: str, date: int, figs: str, preview: bool, verbose: bool, prompt_user: bool):
    # TODO: docs

    # Interpret --figs flag and return a list of figure names to generate
    fig_list = interpret_figs(figs, report_type)
    print(f"Generating the following figures: {fig_list}")

    # Clone the template aprx to a temporary directory
    temp_dir = tempfile.mkdtemp()
    working_aprx = os.path.join(temp_dir, "working_aprx.aprx")
    shutil.copyfile(template_aprx, working_aprx)
    aprx = arcpy.mp.ArcGISProject(working_aprx)  # Open the working aprx in ArcPy
    for fig_id in fig_list:
        print(f"Generating maps for fig {fig_id}...")
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
                label = layer_info["label"]

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
                new_layer_path = find_layer_file(date, layer_info, prompt_user=prompt_user)

                # Special handling for rasters with zero-valued cells
                if new_layer_path.endswith(".tif") and not new_layer_path.endswith(
                        "_nulled.tif") and contains_zero_value_cells(new_layer_path):
                    nulled_path = new_layer_path.replace(".tif", "_nulled.tif")
                    zero_to_no_data(new_layer_path, nulled_path, prompt_user=prompt_user, verbose=verbose)
                    new_layer_path = nulled_path

                # Update labels
                if label == "None" or label == "" or label == [] or not label:
                    # Set the data source
                    _map.addDataFromPath(new_layer_path)

                    # Update layer symbology
                    if file_type in layer_formats:
                        layer = _map.listLayers(f"*{layer_id}*")[0]
                        layer.symbology = symbology

                elif label == "Anno":
                    # Find annotation layer in map
                    # Maybe label:"anno_layer_pattern"
                    # Parse imported layer in python? Can it do dbf?
                    # Set text accordingly
                    pass

                elif isinstance(label, list): # Layer pattern
                    label_pattern = label[0]
                    join_field = label[1]

                    # Create new table view and join it to label layer
                    _map.addTable(arcpy.management.MakeTableView(new_layer_path, f"{layer_id}")[0])
                    label_layers = _map.listLayers(f"*{label_pattern}*")
                    if not label_layers:
                        raise ValueError("No layers matching pattern '*{label_pattern}*' found in '{_map.name}'")
                    if len(label_layers) > 1:
                        if prompt_user:
                            # Ask the user to select which file to use
                            print(f"Multiple layers matching pattern '*{label_pattern}*' found in '{_map.name}'.")
                            for i, file in enumerate(label_layers):
                                print(f"\t{i + 1}. {file}")
                            while True:
                                print(f"Enter a number from 1 to {len(label_layers)}:", end=" ")
                                result = input()
                                try:
                                    result = int(result)
                                    if result in range(1, len(label_layers) + 1): break
                                except ValueError:
                                    pass
                            label_layer = label_layers[int(result) - 1]
                            print(f"Using {label_layer}")
                        else:
                            print(f"Warning: Multiple layers matching pattern '*{label_pattern}*' found in "
                                  f"'{_map.name}'. Using {label_layers[0]}.")
                            label_layer = label_layers[0]
                    else:
                        label_layer = label_layers[0]

                    # Remove existing joins with the label layer
                    try:
                        arcpy.management.RemoveJoin(label_layer)
                    except:
                        pass

                    # Create new join between label_layer and source table
                    arcpy.management.AddJoin(label_layer, join_field, layer_id, join_field)
                    label_layer.showLabels = True
                else:
                    raise Exception(f"Invalid label type '{label}' for layer '{layer_id}' in map '{_map.name}'!")


        # Export the layout to JPEG
        layout = aprx.listLayouts(f"*{fig_id}*")[0]
        if not layout:
            raise ValueError(f"No layout matching '{fig_id}' found in '{working_aprx}'!")
        layout.name = f"{date}_{report_type}_Fig{fig_id}"

        # Change date text
        text_elements = layout.listElements("TEXT_ELEMENT")
        for element in text_elements:
            if "date" in element.name.lower():
                if "date_modis" in element.name.lower():
                    modis_date = get_modis_date(date)
                    formatted_date = datetime.strptime(str(modis_date), "%Y%m%d").strftime("%B %d, %Y")
                    pass
                elif "date_noyear" in element.name.lower(): # January 01
                    formatted_date = datetime.strptime(str(date), "%Y%m%d").strftime("%B %d")
                else: # January 01, 2000
                    formatted_date = datetime.strptime(str(date), "%Y%m%d").strftime("%B %d, %Y")

                element.text = f"{formatted_date}"

        output_dir = os.path.join(output_parent_dir, f"{date}_{report_type}_JPEGmaps")
        os.makedirs(output_dir, exist_ok=True)
        layout.exportToJPEG(os.path.join(output_dir, layout.name + ".jpg"), resolution=300)
        print(f"Finished generating maps for fig {fig_id}.")

    # Clean up
    del aprx

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

    # Generate each figure as specified by --figs
    generate_maps(args.report_type, args.date, args.figs, args.preview, args.verbose, args.prompt_user)

    # Crop the whitespace of each generated map
    output_dir = os.path.join(output_parent_dir, f"{args.date}_{args.report_type}_JPEGmaps")
    for fig_id in interpret_figs(args.figs, args.report_type):
        matches = glob.glob(os.path.join(output_dir, f"*{fig_id}.jpg"))
        if not matches:
            raise FileNotFoundError(f"No file matching pattern '*{fig_id}' found in '{output_dir}'!")
        jpeg_map = matches[0]
        print(f"Cropping {jpeg_map} ...")
        crop_whitespace(jpeg_map)

    # TODO: add support for preview flag

if __name__ == '__main__':
    main()