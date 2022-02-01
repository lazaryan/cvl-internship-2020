#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script file for visualization of source data (image, mask and image + mask)
"""

import const
import cv2
import os

import configargparse

from utils.fs import create_directory


def parse_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)

    parser.add_argument('--save', help='Path to save creating images')
    parser.add_argument('--name_image', help='Name image for view', default='1_1.tif')
    parser.add_argument('--count', help='The number of images displayed on the screen', default=0)
    parser.add_argument('--dir_images', help='Path to dir, where saving images')
    parser.add_argument('--border_color', help='Nerve Stroke Color. Example: 255 255 23', default=const.COLOR_CONTOUR)
    parser.add_argument('--border_size', help='Nerve Stroke Line Width', type=int, default=const.SIZE_CONTOUR)

    return parser.parse_args()


def save_image(image_name, image, path_to_dir):
    """
    Function for saving image
    Args:
        image_name: type=String, name saving file
        image: type=ArrayOfArray, saving file
        path_to_dir: type=String, path to dir for saving image

    Returns:
        type=Bool, true if there were no errors
    """
    if not os.path.isdir(path_to_dir):
        create_directory(path_to_dir)

    cv2.imwrite(os.path.join(path_to_dir, image_name), image)

    return True


def show_image(image_name, path_to_dir):
    """
    Function for show image + mask
    Args:
        image_name: type=String, name image
        path_to_dir: type=String, path to the directory where the images are located

    Returns:
        type=Bool, true if there were no errors
    """
    args = parse_args()

    # Get image and mask
    img = cv2.imread(path_to_dir + image_name)
    # TODO: total regular degeneration (excluding file format)
    mask = cv2.imread((path_to_dir + image_name).replace('.tif', '_mask.tif'))  # add prefix mask
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # convert to gray

    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Get contour

    color_border = args.border_color.split(' ')
    color_border.reverse()
    color_border = tuple(int(channel) for channel in color_border)

    # Show mask and image+mask if contour found
    if contour[0]:
        image_and_contour = img.copy()
        cv2.drawContours(image_and_contour, contour[0], -1, color_border, int(args.border_size))

        # Save the resulting image if the front path
        if args.save:
            save_image(image_name, image_and_contour, args.save)

        cv2.imshow("Mask: " + image_name, mask)
        cv2.imshow("Nerve contour: " + image_name, image_and_contour)

    # Show images
    cv2.imshow("Photo: " + image_name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return True


def main():
    """Application entry point."""
    args = parse_args()

    dir_images = args.dir_images if args.dir_images is not None else os.path.join(const.PATH_TO_DATASET, 'train/')

    if args.count:
        count = 0
        for _, _, files in os.walk(dir_images):
            for filename in files:
                if filename.find('mask') == -1:
                    show_image(filename, dir_images)
                    count = count + 1

                if count == args.count:
                    break
    else:
        show_image(args.name_image, dir_images)


if __name__ == '__main__':
    main()
