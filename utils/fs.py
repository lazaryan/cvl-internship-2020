"""
File containing file system functions
"""

import os
import errno
import shutil


def create_directory(path=''):
    """
    Function for create directory
    Args:
        path: path to dir

    Returns:
        type=Bool, true if there were no errors
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exception:
        if exception != errno.EEXIST:
            raise

    return True


def clear_directory(path):
    """
    Function for clear directory
    Args:
        path: path to directory

    """
    for root, dirs, files in os.walk(path):
        for file in files:
            os.unlink(os.path.join(root, file))
        for directory in dirs:
            shutil.rmtree(os.path.join(root, directory))
