#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for view working network
"""

import segmentation_models_pytorch as smp
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

import configargparse

from train.dataset import SegmentDataset
from common.augmentation import (get_preprocessing, get_validation_augmentation)


def parse_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__, default_config_files=['./demo.cfg'])

    parser.add_argument('--model_path', help='Path to model file', required=True)
    parser.add_argument('--dir_images', help='Path to images for visualize', required=True)
    parser.add_argument('--count', help='How images visualize', type=int, default=1)

    return parser.parse_args()


def visualize(**images):
    """Plot images in one row"""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap='gray')
    plt.show()


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

    dataset = SegmentDataset(
        args.dir_images + 'img/',
        args.dir_images + 'mask/',
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    for i in range(args.count):
        n = np.random.choice(len(dataset))

        image, gt_mask = dataset[n]
        image_vis = image - image.min()
        image_vis /= image_vis.max()
        image_vis = (image_vis * 255).astype('uint8')
        gt_mask = gt_mask.astype('uint8')

        image_vis = image_vis.transpose(1, 2, 0)
        gt_mask = gt_mask.transpose(1, 2, 0)

        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2GRAY)

        x_tensor = torch.from_numpy(image).to('cuda').unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        pr_mask = pr_mask.squeeze().cpu().numpy()

        visualize(
            image=image_vis,
            ground_truth_mask=gt_mask,
            predicted_mask=pr_mask
        )


if __name__ == '__main__':
    main()
