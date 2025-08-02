"""
Library containing utility functions

**Functions:**
    - ``confirm_process``: Confirms with the user if the current process should continue, and exits if not
    - ``copy_files``: Copies a list of files to a directory
    - ``delete_files``: Deletes all files from a list

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
