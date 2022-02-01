#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for converting data (change format)
"""
import os
import errno
import numpy as np
import re
import h5py

import configargparse

from utils.fs import create_directory


def parse_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)

    parser.add_argument('--dir-inp', help='Path to dir, where saving images', type=str, required=True)
    parser.add_argument('--dir-out', help='The path to the dir where to save images', type=str, required=True)
    parser.add_argument('--format', help='Format for converting', choices=['hdf5'], default='hdf5')

    return parser.parse_args()


def convert_to_hdf5(dir_inp, filename, dir_out):
    """
    Convert file to HDF5 format
    Args:
        dir_inp: type=String, Path to dir, where saving images
        filename: type=String, name file
        dir_out: type=String, The path to the dir where to save images

    Returns:

    """
    path_to_file = dir_inp + filename

    # Get data for new file
    fin = open(path_to_file, 'rb')
    binary_data = fin.read()
    name_without_format = re.search(r'(.*)\..*', filename).groups()[0]
    path_to_output_file = dir_out + name_without_format + '.hdf5'

    # Create and saving new file
    file = h5py.File(path_to_output_file)
    data_type = h5py.special_dtype(vlen=np.dtype('uint8'))
    data_set = file.create_dataset('binary_data', (100,), dtype=data_type)
    data_set[0] = np.fromstring(binary_data, dtype='uint8')

    fin.close()


def main():
    args = parse_args()

    if not os.path.isdir(args.dir_out):
        create_directory(args.dir_out)

    for root, _, files in os.walk(args.dir_inp):
        for filename in files:
            if args.format == 'hdf5':
                convert_to_hdf5(root, filename, args.dir_out)


if __name__ == '__main__':
    main()
