#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for displaying statistics on source images

Doing:
    - display on how many images no nerve was found
    - display on how many images a nerve was found
    - print the average number of pixels occupied by a nerve
"""

import os
import cv2

import argparse
import matplotlib.pyplot as plt

import const


def parse_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--dir', help='Path to dir, where saving images', type=str)

    return parser.parse_args()


def find_nerve(mask):
    """
    Function for checking the existence of a nerve in an image
    Args:
        mask: type=File, image for checking

    Returns:
        type=Bool, True if nerve found
    """
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contour[0]


def get_size_nerve(contour):
    """
    The function of obtaining the size (in px) of a nerve according to its contour
    Args:
        contour: type=ArrayOfArray, contour nerve

    Returns:
        type=int, count px
    """
    return cv2.contourArea(contour)


def draw_statistic(result):
    """
    Function for rendering statistics
    Args:
        result: type=dict:
            not_nerve: type=int, count images, where nerve not found
            all: type=int, count all images
            nerves: type=ArrayOfArray, area nerve and area photo

    Returns:

    """
    data = {
        f'count not found nerve ({result["not_nerve"]})': result['not_nerve'],
        f'count founding nerves ({ result["all"] - result["not_nerve"] })': result['all'] - result['not_nerve']
    }

    group_data = list(data.values())
    group_names = list(data.keys())

    fig, axs = plt.subplots(2, 1, figsize=(15, 7))

    # draw statistic
    axs[0].bar(group_names, group_data)
    axs[0].set(title=f'Statistic found nerves, all ({result["all"]})', ylabel='Count')

    # draw graphics
    share_image = [(el[0]/el[1]) * 100 for el in result['nerves']]

    axs[1].grid(True)
    axs[1].plot(range(len(result['nerves'])), share_image, 'ro')
    axs[1].set(title='Percentage of Nerve Per Image', ylabel='%')
    axs[1].set_yticks(range(0, 10, 2))

    plt.show()


def main():
    """Application entry point."""
    args = parse_args()

    dir_images = args.dir if args.dir is not None else os.path.join(const.PATH_TO_DATASET, 'train/')

    result = {'not_nerve': 0, 'all': 0, 'nerves': []}

    for _, _, files in os.walk(dir_images):
        for filename in files:
            if filename.find('mask') != -1:
                result['all'] = result['all'] + 1

                mask = cv2.imread(dir_images + filename)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

                height, width = mask.shape[:2]

                nerve = find_nerve(mask)

                if not nerve:
                    result['not_nerve'] = result['not_nerve'] + 1
                else:
                    result['nerves'].append([get_size_nerve(nerve[0]), height * width])

    draw_statistic(result)


if __name__ == '__main__':
    main()
