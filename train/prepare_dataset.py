#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The script for preparing the source data for training the network
"""

import os
import shutil
import glob

import configargparse

import convert
import const
from utils.fs import (create_directory, clear_directory)


def parse_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__, default_config_files=['./prepare_config.cfg'])

    parser.add_argument('--config', is_config_file=True, help='Path to config file')
    parser.add_argument('--clear', help='clear directory', action='store_true')
    parser.add_argument('--percent_train', help='percentage of images that will be used to train the network',
                        type=float, default=0.8)
    parser.add_argument('--convert', help='format to convert images to')
    parser.add_argument('--dir_train', required=True, help='path to save images for train net')
    parser.add_argument('--dir_valid', required=True, help='path to save images for validation net')

    return parser.parse_args()


def main():
    """Application entry point."""
    args = parse_args()

    dir_images = f'../{const.DIR_DATA}{const.NAME_PROJECT}/{const.DIR_TRAIN}'
    dir_saving = args.dir_train

    files = glob.glob1(dir_images, "*[1-9].tif")

    count = int((len(files) * float(args.percent_train)))
    count_save = 0

    # clear directory for saving images
    if args.clear:
        clear_directory(const.ROOT + '/' + args.dir_train)
        clear_directory(const.ROOT + '/' + args.dir_valid)

    for filename in files:
        count_save += 1
        mask_name = filename.replace('.t', '_mask.t')  # Get name mask file

        if count_save == count:
            dir_saving = args.dir_valid

        # create directory if they are not
        if not os.path.isdir(dir_saving + '/img'):
            create_directory(dir_saving + '/img')

        if not os.path.isdir(dir_saving + '/mask'):
            create_directory(dir_saving + '/mask')

        if args.convert == 'hdf5':
            convert.convert_to_hdf5(dir_images, filename, dir_saving + '/img/')
            convert.convert_to_hdf5(dir_images, mask_name, dir_saving + '/mask/')
        else:
            shutil.copy2(f'{dir_images}/{filename}', f'{dir_saving}/img/')
            shutil.copy2(f'{dir_images}/{mask_name}', f'{dir_saving}/mask/{filename}')


if __name__ == '__main__':
    main()
