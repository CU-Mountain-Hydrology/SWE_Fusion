"""
Replaces all zero-valued cells in a raster with NoData

**Usage:**
    python zero_to_no_data.py input_raster [output_raster] [--verbose]

**Arguments:**
    - ``input_raster``: Filepath of the raster to remove zero-valued cells from

**Options:**
    - ``output_raster``: Filepath of the output raster. Default: input_raster
    - ``--verbose``, ``-v``: Enable verbose output messages
"""

from utils import confirm_process, copy_files, delete_files
import arcpy
from arcpy.sa import * # Spatial Analyst module
import argparse
import os


def find_raster_files(path: str, verbose = True, warn = True) -> list[str]:
    """
    Finds all common ArcGIS Pro files associated with a raster dataset

    :param path: Path to the main raster file (.tif)
    :type path: str
    :param verbose: Enable verbose output messages. Default: True
    :type verbose: bool
    :param warn: Enable warning messages. Default: True
    :type warn: bool
    """
    if not os.path.exists(path):
        if warn: print(f"find_raster_files warning: Raster file does not exist: {path}")
        return []

    # Strip .tif extension from filepath
    base, _ = os.path.splitext(path)

    known_extensions = [
        ".tif",         # The main file itself
        ".aux",         # Generic auxiliary
        ".aux.xml",     # Metadata XML
        ".ovr",         # Overviews
        ".xml",         # Metadata XML
        ".tfw",         # World file for TIFF
        ".rrd",         # Reduced resolution dataset (ERS/IMG)
        ".vat.dbf",     # Value Attribute Table
        ".clr",         # Color map
        ".rrd.xml",     # Additional ERS metadata
        ".hdr",         # Header files for binary rasters
        ".prj",         # Projection file
        ".wld",         # Generic world file
    ]

    related_files = []
    for extension in known_extensions:
        file = base + extension
        if os.path.exists(file):
            if verbose: print(f"Found raster file: {file}")
            related_files.append(file)

    if len(related_files) == 0:
        if warn: print(f"find_raster_files warning: unable to find any files associated with the raster file: {path}")

    return related_files


def contains_zero_value_cells(raster_filepath: str, verbose = True) -> bool:
    """
    Checks whether a raster file contains any cells with a value of 0.

    :param raster_filepath: Path to the raster file to be checked.
    :type raster_filepath: str
    :param verbose: Whether to print progress messages. Defaults to True.
    :type verbose: bool
    :return: True if the raster contains any zero-value cells, False otherwise.
    :rtype: bool
    """
    # TODO: Optimize with NumPy or by calculating statistics
    if verbose: print(f"Checking for cells with value 0 in {repr(raster_filepath)}...")

    arcpy.CheckOutExtension("Spatial")
    try:
        raster = Raster(raster_filepath)
        zero_mask = Con(raster == 0, 1)
        has_zero = int(arcpy.management.GetCount(zero_mask)[0]) > 0

        if verbose:
            print("Cells with value 0 found!" if has_zero else "No cells with value 0 found!")

        return has_zero
    finally:
        # Always check the extension back in, even on crash or error. ArcPy does not handle this automatically.
        arcpy.CheckInExtension("Spatial")

def zero_to_no_data(input_raster_filepath: str, output_raster_filepath: str = None, prompt_user = True, verbose = True) -> None:
    """
    Converts all raster cells with value 0 to NoData

    :param input_raster_filepath: Path to the raster file
    :param output_raster_filepath: Optional output raster file path. Default: input_raster_filepath
    :param prompt_user: Enable prompting the user before overwriting files. Default: True
    :param verbose: Enable verbose output messages. Default: True
    :return:
    """

    # Check input raster exists
    if not os.path.exists(input_raster_filepath):
        print("Input raster file does not exist. Aborting.")
        exit(1)

    overwrite_input = input_raster_filepath == output_raster_filepath or output_raster_filepath is None

    # Prompt user if any files are to be overwritten
    if output_raster_filepath is None:
        if prompt_user and confirm_process("No output raster specified, the input raster will be overwritten!"):
            print("Overwriting input raster...")
        output_raster_filepath = input_raster_filepath
    elif input_raster_filepath == output_raster_filepath:
        if prompt_user and confirm_process("The input raster is the same as the output raster and will be overwritten!"):
            print("Overwriting input raster...")
    elif os.path.exists(output_raster_filepath):
        if prompt_user and confirm_process("The output raster already exists and will be overwritten!"):
            print("Overwriting output raster...")
        delete_files(find_raster_files(output_raster_filepath,verbose=verbose),verbose=verbose)

    arcpy.CheckOutExtension("Spatial")
    try:
        # Create raster and set cells with value 0 to NoData
        input_raster = Raster(input_raster_filepath)
        output_raster = SetNull(input_raster == 0, input_raster)

        if not overwrite_input:
            # Delete the existing output file if it exists, then save the new raster
            delete_files(find_raster_files(output_raster_filepath,verbose=verbose,warn=False),verbose=verbose)
            output_raster.save(output_raster_filepath)
            if verbose: print("Converted zeros to NoData and saved:", output_raster_filepath)
            del input_raster, output_raster
            return

        # ArcPy struggles when trying to overwrite the input. Instead, we save to a temp file then copy it back.
        temp_dir = os.getenv("TEMP")
        temp_tif_path = os.path.join(temp_dir, os.path.basename(output_raster_filepath))
        if os.path.exists(temp_tif_path):
            os.remove(temp_tif_path)
        output_raster.save(temp_tif_path)

        # Remove file locks so they can be deleted
        del input_raster, output_raster
        delete_files(find_raster_files(output_raster_filepath, verbose=verbose), verbose=verbose)

        # Copy temporary files back to the output raster directory
        temp_files = find_raster_files(temp_tif_path, verbose=verbose)
        copy_files(temp_files, os.path.dirname(output_raster_filepath), verbose=verbose)
        delete_files(temp_files, verbose=verbose)

        # Delete OVR file as it causes rendering problems
        ovr_file = output_raster_filepath + ".ovr"
        if os.path.exists(ovr_file):
            delete_files([ovr_file], verbose=verbose)

    finally:
        # Always check the extension back in, even on crash or error. ArcPy does not handle this automatically.
        arcpy.CheckInExtension("Spatial")


def main():
    # Parse command line arguments to get input and output raster locations
    parser = argparse.ArgumentParser()
    parser.add_argument("input_raster", type=str, help="File path to the raster to process")
    parser.add_argument("output_raster", type=str, nargs="?", help="File path for the processed output raster")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output messages")
    args = parser.parse_args()

    # Check if the input raster contains zero-value cells
    if not contains_zero_value_cells(args.input_raster, verbose=args.verbose):
        print(f"No cells with value 0 found in input raster: {args.input_raster}")
        exit(1)

    # Convert cells with value 0 to NoData
    zero_to_no_data(args.input_raster, args.output_raster, prompt_user=True, verbose=args.verbose)

if __name__ == '__main__':
    main()