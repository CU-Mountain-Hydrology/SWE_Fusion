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
ww_aprx = "U:\EricG\MapTemplate\MapTemplate.aprx"       # Project containing template for each figure
snm_aprx = "U:\EricG\MapTemplate\SNM_Template.aprx"
# product_source_dir = r"U:\EricG\testing_Directory"    # Parent directory of the YYYYMMDD_RT_Report folders
product_source_dir = r"W:\documents\2026_RT_Reports"    # Parent directory of the YYYYMMDD_RT_Report folders
# TODO: separate source & output dirs for WW and SNM
# snm_source_dir = r"U:\EricG\testing_Directory\SNM"
snm_source_dir = r"J:\paperwork\0_UCSB_DWR_Project\2026_RT_Reports"
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
                {"layer": "p8", "format": "tif", "dir": "*UseThis*", "label": "None"}
            ]
        }
    },
    "1b": {
        "maps": {
            "1b": [
                {"layer": "anomRegion_table", "format": "csv", "dir": "*UseAvg", "label": ["WW_Regions_albn83_label","ZONE_CODE"]}
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
                {"layer": "p8", "format": "tif", "dir": "*UseThis*", "label": "None"}
            ],
            "3b": [
                {"layer": "anom0_200_msk", "format": "tif", "dir": "*UseAvg", "label": "None"}
            ],
            "3c": [
                {"layer": "p9", "format": "tif", "dir": "*UseAvg", "label": "None"},
                {"layer": "anom_table_save", "format": "dbf", "dir": "*UseAvg", "label": ["*_Basins_albn83", "SrtName"]},
            ],
            "3d": []
        }
    },
    "4": {
        "maps": {
            "4a": [
                {"layer": "p8", "format": "tif", "dir": "*UseThis*", "label": "None"},
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
                {"layer": "p8", "format": "tif", "dir": "*UseThis*", "label": "None"}
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
                {"layer": "p8", "format": "tif", "dir": "*UseThis*", "label": "None"}
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
                {"layer": "anomRegion_table", "format": "dbf", "dir": "*UseAvg", "label": ["dwr_region_poly_labels", "Regions"]}
            ]
        }
    },
    "1": {
        "maps": {
            "real_time_swe": [
                {"layer": "p8*msk", "format": "tif", "dir": "*UseThis*", "label": "None"}
            ],
            "anomaly": [
                {"layer": "anom0_200_msk", "format": "tif", "dir": "*UseAvg", "label": "None"}
            ],
            "watershed_map": [
                {"layer": "p9", "format": "tif", "dir": "*UseAvg", "label": "None"},
                {"layer": "anom_table_save", "format": "dbf", "dir": "*UseAvg", "label": ["dwr_basins_geon83", "SrtName"]},
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
                {"layer": "p8*msk", "format": "tif", "dir": "*UseThis*", "label": "None"}
            ]
        }
    },
    "5": {
        "maps": {
            "TC_MODIS_image": [
                {"layer": "snapshot*UseThis", "format": "tif", "dir": "MODIS", "label": "None"}
            ]
        }
    },
    "6": {
        "maps": {
            "SNODAS_swe": [
                {"layer": "Cp_m_albn83_clp", "format": "tif", "dir": "SNODAS", "label": "None"}
            ],
            "SNODAS_diff": [
                {"layer": "SNODAS_Regress", "format": "tif", "dir": "*UseThis*", "label": "None"}
            ],
            "SNODAS_swe_overlap": [
                {"layer": "both", "format": "tif", "dir": "SNODAS", "label": "None"}
            ]
        }
    },
    "7": {
        "maps": {
            "mean_swe": [
                {"layer": "mean_msk", "format": "tif", "dir": "*UseThis*", "label": "None"},
                {"layer": "Zero_sensors", "format": "shp", "dir": "", "label": "None"},
                {"layer": "SNM_*_sensors", "format": "shp", "dir": "", "label": "None"},
                # {"layer": "Zero_CCR", "format": "shp", "dir": "", "label": "None"},
                {"layer": "CCR_sensors", "format": "shp", "dir": "", "label": "None"},
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
import uuid
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
        case 'SNM':
            all_figs = set(snm_fig_data.keys())
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


def find_layer_file(report_type: str, date: int, layer_info: dict, prompt_user = True, warn = True) -> str:
    """
    Finds the specific layer file to use

    :param report_type: Type of report to interpret figures for. e.g., WW, SNM
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
    # TODO: copy files out of results dir to avoid corruption problems. Automate this process
    if report_type == "WW":
        rt_report_dir = os.path.join(product_source_dir, str(date) + "_RT_report_ET")
    else: # SNM
        rt_report_dir = os.path.join(snm_source_dir, str(date) + "_RT_report_ET") # TODO: config
        # rt_report_dir = os.path.join(snm_source_dir, str(date) + "_RT_Report")

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
                layer_file = file
                if warn: print(f"find_layer_file warning: Multiple files matching pattern '{layer_id}' found! Using {layer_file}")
                break
            elif layer_id == "anom_table_save" and file.endswith(f"{date}anom_table_save.dbf"):
                layer_file = file
                if warn: print(f"find_layer_file warning: Multiple files matching pattern '{layer_id}' found! Using {layer_file}")
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


def get_modis_date(layer_file) -> int:
    # TODO: docs
    match = re.search(r"snapshot-(\d{4})-(\d{2})-(\d{2})", layer_file)
    if not match:
        raise ValueError("No valid MODIS snapshot date found in path: ", layer_file)

    year, month, day = match.groups()
    return int(f"{year}{month}{day}")


def generate_maps(report_type: str, date: int, figs: str, preview: bool, verbose: bool, prompt_user: bool):
    # TODO: docs

    # Interpret --figs flag and return a list of figure names to generate
    fig_list = interpret_figs(figs, report_type)
    print(f"Generating the following figures: {fig_list}")

    # Clone the template aprx to a temporary directory
    if report_type == "WW":
        template_aprx = ww_aprx
        fig_data_dict = ww_fig_data
    else:
        template_aprx = snm_aprx
        fig_data_dict = snm_fig_data

    temp_dir = tempfile.mkdtemp()
    working_aprx = os.path.join(temp_dir, "working_aprx.aprx")
    shutil.copyfile(template_aprx, working_aprx)
    aprx = arcpy.mp.ArcGISProject(working_aprx)  # Open the working aprx in ArcPy
    for fig_id in fig_list:
        print(f"Generating maps for fig {fig_id}...")
        fig_data = fig_data_dict.get(fig_id)
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
                ref_layer = None

                if file_type in layer_formats:
                    undefined_layer = _map.listLayers(f"*{layer_id}*")[0]
                    symbology = undefined_layer.symbology

                    # Save position information before removing
                    all_layers = _map.listLayers()
                    layer_index = all_layers.index(undefined_layer)
                    # Get reference layer (the one just above the current layer)
                    ref_layer = all_layers[layer_index - 1] if layer_index > 0 else None

                    _map.removeLayer(undefined_layer)
                elif file_type in table_formats:
                    undefined_table = _map.listTables(f"*{layer_id}*")[0]
                    _map.removeTable(undefined_table)

                # Find the new layer source
                new_layer_path = find_layer_file(report_type, date, layer_info, prompt_user=prompt_user)

                # Handle MODIS date
                if "snapshot" in new_layer_path:
                    modis_date = get_modis_date(new_layer_path)

                # Special handling for non-MODIS rasters with zero-valued cells
                if new_layer_path.endswith(".tif") and not new_layer_path.endswith(
                        "_nulled.tif") and not "snapshot" in new_layer_path and contains_zero_value_cells(new_layer_path):
                    nulled_path = new_layer_path.replace(".tif", "_nulled.tif")
                    zero_to_no_data(new_layer_path, nulled_path, prompt_user=prompt_user, verbose=verbose)
                    new_layer_path = nulled_path

                # Update labels
                if label == "None" or label == "" or label == [] or not label:
                    if file_type in layer_formats:
                        # Create layer object
                        if file_type == "shp":
                            new_layer = arcpy.management.MakeFeatureLayer(new_layer_path, f"temp_{layer_id}").getOutput(0)
                        else:  # raster (tif)
                            new_layer = arcpy.management.MakeRasterLayer(new_layer_path, f"temp_{layer_id}").getOutput(0)

                        # Insert at saved position
                        if ref_layer:
                            _map.insertLayer(ref_layer, new_layer, "AFTER")
                        else:
                            _map.addLayer(new_layer, "TOP")

                        # Update symbology
                        layer = _map.listLayers(f"*{layer_id}*")[0]
                        if "snapshot" not in new_layer_path and symbology:
                            layer.symbology = symbology
                    else:  # table
                        _map.addDataFromPath(new_layer_path)
                elif isinstance(label, list):  # Join table to label layer
                    label_pattern = label[0]  # Pattern of the shp layer with labels enabled
                    join_field = label[1]  # Field in both the label layer and the join table

                    # Find label layer first
                    label_layers = _map.listLayers(f"*{label_pattern}*")
                    if not label_layers:
                        raise ValueError(f"No layers matching pattern '*{label_pattern}*' found in '{_map.name}'")

                    if len(label_layers) > 1:
                        if prompt_user:
                            print(f"Multiple layers matching pattern '*{label_pattern}*' found in '{_map.name}'.")
                            for i, lyr in enumerate(label_layers):
                                print(f"\t{i + 1}. {lyr.name}")
                            while True:
                                print(f"Enter a number from 1 to {len(label_layers)}:", end=" ")
                                result = input()
                                try:
                                    result = int(result)
                                    if result in range(1, len(label_layers) + 1):
                                        break
                                except ValueError:
                                    pass
                            label_layer = label_layers[int(result) - 1]
                            print(f"Using {label_layer.name}")
                        else:
                            print(f"Warning: Multiple layers matching pattern '*{label_pattern}*' found. "
                                  f"Using {label_layers[0].name}.")
                            label_layer = label_layers[0]
                    else:
                        label_layer = label_layers[0]

                    # Remove existing joins from the original layer
                    try:
                        arcpy.management.RemoveJoin(label_layer)
                    except:
                        pass

                    # Save label expression for later
                    original_label_expression = label_layer.listLabelClasses()[0].expression
                    if verbose:
                        print(f"Original label expression: {original_label_expression}")

                    # Create a table view
                    table_view_name = f"temp_table_{uuid.uuid4().hex[:8]}"
                    arcpy.management.MakeTableView(new_layer_path, table_view_name)

                    # Verify join field exists in both
                    if verbose:
                        print(f"Label layer: {label_layer}")
                        print(f"Label layer type: {type(label_layer)}")
                        print(f"Label layer name: {label_layer.name}")
                        print(f"Label layer dataSource: {label_layer.dataSource}")

                        label_data_source = label_layer.dataSource
                        if not os.path.exists(label_data_source):
                            label_data_source = label_data_source + '.shp'
                        label_fields = [f.name for f in arcpy.ListFields(label_data_source)]
                        table_fields = [f.name for f in arcpy.ListFields(table_view_name)]
                        print(f"Feature layer fields: {label_fields}")
                        print(f"Table fields: {table_fields}")
                        print(f"Join field: {join_field}")

                        if join_field not in label_fields:
                            raise ValueError(f"Join field '{join_field}' not found in feature layer")
                        if join_field not in table_fields:
                            raise ValueError(f"Join field '{join_field}' not found in table")

                    # Perform the join directly on the map layer
                    if verbose: print(
                        f"Attempting join: layer={label_layer.name}, field={join_field}, table={table_view_name}")
                    arcpy.management.AddJoin(
                        in_layer_or_view=label_layer,
                        in_field=join_field,
                        join_table=table_view_name,
                        join_field=join_field,
                        join_type="KEEP_ALL"
                    )

                    if verbose:
                        joined_fields = [f.name for f in arcpy.ListFields(label_layer)]
                        print(f"Fields after join: {joined_fields}")

                    # Find the actual table prefix used in the join (extract from filename)
                    table_path = os.path.basename(new_layer_path)
                    if table_path.lower().endswith('.csv'):
                        # Keep the .csv extension for CSV files
                        table_filename = table_path
                    else:
                        # Remove extension for other file types (like .dbf)
                        table_filename = os.path.splitext(table_path)[0]

                    if verbose:
                        print(f"Table filename (used as prefix): {table_filename}")

                    # Set the label expression to use the joined field
                    for lbl_class in label_layer.listLabelClasses():
                        # After joining, field references need to include the table prefix
                        # Replace $feature.fieldname with $feature['tablename.fieldname']
                        import re

                        # Extract field name from the original expression (e.g., "Average" from "$feature.Average")
                        # This regex finds patterns like $feature.fieldname
                        def replace_field_reference(match):
                            field_name = match.group(1)
                            return f"$feature['{table_filename}.{field_name}']"

                        new_expression = re.sub(r'\$feature\.(\w+)', replace_field_reference, original_label_expression)

                        lbl_class.expression = new_expression
                        if verbose:
                            print(f"Updated label expression: {new_expression}")

                    # Ensure layer is visible with labels
                    label_layer.visible = True
                    label_layer.showLabels = True

                    if verbose:
                        print(
                            f"Layer '{label_layer.name}' visible: {label_layer.visible}, showLabels: {label_layer.showLabels}")

                    # Save the project
                    aprx.save()

                    # Clean up (non-critical)
                    try:
                        arcpy.management.Delete(table_view_name)
                    except:
                        pass

        # Export the layout to JPEG
        layout = aprx.listLayouts(f"*{fig_id}*")[0]
        if not layout:
            raise ValueError(f"No layout matching '{fig_id}' found in '{working_aprx}'!")
        layout.name = f"{date}_{report_type}_Fig{fig_id}"

        # Change date text
        text_elements = layout.listElements("TEXT_ELEMENT")
        for element in text_elements:
            if "date" in element.name.lower():
                if "date_modis" in element.name.lower() and modis_date:
                    formatted_date = datetime.strptime(str(modis_date), "%Y%m%d").strftime("%B %#d, %Y")
                elif "date_noyear" in element.name.lower(): # January 01
                    formatted_date = datetime.strptime(str(date), "%Y%m%d").strftime("%B %#d")
                else: # January 01, 2000
                    formatted_date = datetime.strptime(str(date), "%Y%m%d").strftime("%B %#d, %Y")
                element.text = f"{formatted_date}"

            elif "pctavg" in element.name.lower():
                date_str = str(date)
                formatted_date = f"{int(date_str[4:6])}/{int(date_str[6:8])}" # 20250331 => 3/31
                element.text = element.text.replace("3/31", formatted_date)

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