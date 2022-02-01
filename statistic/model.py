#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for view working network
"""
import os

import segmentation_models_pytorch as smp
import torch
import cv2

import argparse

from common.augmentation import (get_preprocessing, get_validation_augmentation)


def parse_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--model_path', help='Path to model file', required=True)
    parser.add_argument('--dir_images', help='Path to images for visualize', required=True)

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


def get_augmentation_image_and_mask(
        path_to_image,
        path_to_mask,
        preprocessing_fn
):
    """
    Function for augmentation image and mask
    Args:
        path_to_image: path to image
        path_to_mask: path to mask
        preprocessing_fn: function for preprocessing convert

    Returns:
        augmentations image and mask
    """
    image = cv2.imread(path_to_image)
    mask = cv2.imread(path_to_mask)

    augmentation_func = get_validation_augmentation()
    preprocess_fn = get_preprocessing(preprocessing_fn)

    sample = augmentation_func(image=image, mask=mask)
    image, mask = sample['image'], sample['mask']

    image = preprocess_fn(image=image)['image']
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    return image, mask


def get_predict_mask(model, image):
    """
    Function for getting predict mask (using model)
    Args:
        model: Model
        image: initial image

    Returns:
        new image (numpy array)
    """
    x_tensor = torch.from_numpy(image).to('cuda').unsqueeze(0)
    pr_mask = model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    pr_mask = (pr_mask * 255).astype('uint8')

    return pr_mask


def main():
    """Application entry point."""
    args = parse_args()

    result = {
        'init': {
            'all': 0,
            'have nerve': 0,
            'not have nerve': 0,
        },
        'predict': {
            'detect nerve': 0,
            'not detect nerve': 0,
            'detect more nerves': 0,
            'detect lack of nerve': 0,
            'not detect lack of nerve': 0,
        }
    }

    checkpoint = torch.load(args.model_path)

    model = smp.Unet(
        encoder_name=checkpoint['encoder_name'],
        encoder_weights=checkpoint['encoder_weights'],
        classes=checkpoint['classes'],
        activation=checkpoint['activation'],
    )

    model.load_state_dict(checkpoint['model_state_dict'])

    if checkpoint['device'] == torch.device('cuda') and torch.cuda.is_available():
        model.cuda()

    preprocessing_fn = smp.encoders.get_preprocessing_fn(checkpoint['encoder_name'],
                                                         pretrained=checkpoint['encoder_weights'])

    for root, _, files in os.walk(os.path.join(args.dir_images, 'img')):
        result['init']['all'] = count_images = len(files)
        i = 0

        for filename in files:
            i += 1
            print(f'Progress:{i} / {count_images}')

            image, mask = get_augmentation_image_and_mask(
                path_to_image=os.path.join(root, filename),
                path_to_mask=os.path.join(args.dir_images, 'mask', filename),
                preprocessing_fn=preprocessing_fn
            )

            pr_mask = get_predict_mask(model=model, image=image)

            height, width = mask.shape[:2]
            pr_height, pr_width = pr_mask.shape[:2]

            nerve = find_nerve(mask)
            pr_nerve = find_nerve(pr_mask)

            if len(nerve) > 0:
                result['init']['have nerve'] += 1

                if len(pr_nerve) > 0:
                    result['predict']['detect nerve'] += 1

                    if len(pr_nerve) > len(nerve):
                        result['predict']['detect more nerves'] += 1
                else:
                    result['predict']['not detect nerve'] += 1
            else:
                result['init']['not have nerve'] += 1

                if len(pr_nerve) == 0:
                    result['predict']['detect lack of nerve'] += 1
                else:
                    result['predict']['not detect lack of nerve'] += 1

        loss = (
                    result['predict']['not detect nerve'] +
                    result['predict']['detect more nerves'] +
                    result['predict']['not detect lack of nerve']
               ) / result['init']['all']

        statistic = {
            'count images': result['init']['all'],
            'found nerve': (result['predict']['detect nerve'] / result['init']['have nerve']) * 100,
            'detect lack of nerve': (result['predict']['detect lack of nerve'] / result['init']['not have nerve'])*100,
            'loss': loss * 100,
            'result': (1 - loss) * 100
        }

        print(statistic)


if __name__ == '__main__':
    main()
