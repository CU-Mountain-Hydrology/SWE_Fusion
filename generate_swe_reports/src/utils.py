"""
Library containing utility functions

**Functions:**
    - ``confirm_process``: Confirms with the user if the current process should continue, and exits if not
    - ``copy_files``: Copies a list of files to a directory
    - ``delete_files``: Deletes all files from a list
    # TODO: crop function docs
    # TODO: merge_csv function docs

"""

def confirm_process(message: str) -> bool:
    """
    Confirms with the user if the current process should continue, and exits if not

    Default: No

    :param message: Message containing what will happen if the process continues
    :return: True if the user gives confirmation
    """
    print(f"{message} Continue? (y/N)")
    while True:
        user_answer = input().strip().lower()
        if user_answer in ["y", "yes"]:
            return True
        elif user_answer in ["", "n", "no"]:
            print(f"Aborting process.")
            exit(1)
        else:
            print("Invalid input. Please enter y or n.")


def copy_files(files: list[str], directory: str, verbose = True, warn = True) -> None:
    """
    Copies a list of files to a directory

    :param files: List of filepaths to copy
    :param directory: Path to the destination directory
    :param verbose: Enable verbose output messages. Default: True
    :param warn: Enable warning messages. Default: True
    :return:
    """
    import os
    import shutil

    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Target path does not exist or is not a directory: {directory}")

    if len(files) == 0:
        if warn: print(f"copy_files warning: List of files is empty.")
        return

    for file in files:
        if not os.path.exists(file):
            if warn: print(f"copy_files warning: File not found: {file}")
            continue

        try:
            shutil.copy(file, directory)
            if verbose: print(f"Copied: {file} -> {directory}")
        except Exception as e:
            print(f"copy_files error: Failed to copy {file} to {directory}: {e}")



def delete_files(files: list[str], verbose=True, warn = True) -> None:
    """
    Deletes all files from a list

    :param files: List of filepaths to delete
    :param verbose: Enable verbose output messages. Default: True
    :param warn: Enable warning messages. Default: True
    :return:
    """
    import os

    for file in files:
        try:
            os.remove(file)
            if verbose: print(f"Deleted file: {file}")
        except OSError as e:
            if warn: print(f"delete_files warning: Could not delete file {file}: {e}")


from PIL import Image, ImageChops
def crop_whitespace(input_filepath: str, output_filepath: str = None) -> None:
    # TODO: docs

    if output_filepath is None:
        output_filepath = input_filepath

    image = Image.open(input_filepath)
    # TODO: error handling
    image = image.convert("RGB")

    background = Image.new("RGB", image.size, (255, 255, 255))
    diff = Image.eval(ImageChops.difference(image, background), lambda px: px > 10)
    bbox = diff.getbbox()

    if bbox:
        cropped = image.crop(bbox)
        cropped.save(output_filepath)
        print(f"Cropped image saved to {output_filepath}")
    else:
        print(f"Image did not contain any whitespace.")


def merge_swe_csv(input_filepath_1: str, input_filepath_2: str, output_filepath: str) -> None:
    """
    # TODO: docs
    """
    with open(input_filepath_1, "r") as csv1, open(input_filepath_2, "r") as csv2:
        lines1 = csv1.readlines()
        lines2 = csv2.readlines()

    # First two rows are headers
    header = lines1[:2]

    data1 = lines1[2:]
    data2 = lines2[2:]

    # Merge
    merged_lines = header + data1 + data2

    # Save merged CSV
    with open(output_filepath, "w") as output:
        output.writelines(merged_lines)

    print(f"Merged SWE table saved to {output_filepath}")