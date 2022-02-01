#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for get mask nerves and saving his
"""
import os

import segmentation_models_pytorch as smp
import torch
import cv2

import argparse

from common.augmentation import (get_preprocessing, get_validation_augmentation)
from utils.fs import clear_directory


def parse_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--dir_images', help='path to dir where saving images for predict', required=True)
    parser.add_argument('--dir_saving', help='path to dir where saving new masks', required=True)
    parser.add_argument('--model_path', help='Path to model file', required=True)
    parser.add_argument('--clear', help='clear directory', action='store_true')
    parser.add_argument('--count', help='How many images need to be processed', type=int)

    return parser.parse_args()


def main():
    """Application entry point."""
    args = parse_args()

    checkpoint = torch.load(args.model_path)

    model = smp.Unet(
        encoder_name=checkpoint['encoder_name'],
        encoder_weights=checkpoint['encoder_weights'],
        classes=checkpoint['classes'],
        activation=checkpoint['activation'],
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()

    preprocessing_fn = smp.encoders.get_preprocessing_fn(checkpoint['encoder_name'],
                                                         pretrained=checkpoint['encoder_weights'])

    if args.clear:
        clear_directory(args.dir_saving)

    for root, _, files in os.walk(args.dir_images):
        # if the number of processed images is not transmitted, save all
        count = len(files) if not args.count else args.count

        for filename in files:
            count -= 1
            image = cv2.imread(root + filename)

            # resize and convert to tensor
            augmentation_func = get_validation_augmentation()
            preprocess_fn = get_preprocessing(preprocessing_fn)

            image = augmentation_func(image=image)['image']
            image = preprocess_fn(image=image)['image']

            x_tensor = torch.from_numpy(image).to('cuda').unsqueeze(0)
            pr_mask = model.predict(x_tensor)
            pr_mask = (pr_mask.squeeze().cpu().numpy().round())

            # save predict mask
            cv2.imwrite(os.path.join(args.dir_saving, filename), pr_mask.astype('uint8') * 255)

            if count <= 0:
                break


if __name__ == '__main__':
    main()
