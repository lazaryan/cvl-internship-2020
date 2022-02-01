"""
This file contains the class for working with images.
This class stores information about images and their masks.
"""

import os

import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class SegmentDataset(Dataset):
    def __init__(
            self,
            images_dir,
            masks_dir,
            augmentation=None,
            preprocessing=None,
    ):
        """
        Initializing a Data Processing Class
        Args:
            images_dir: type=String, path to images dir
            masks_dir: type=String, path to masks dir for images
            augmentation: type=Function, function for update dataset
            preprocessing: type=Function, Image preprocessing function
        """
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        """
        Retrieving an image and its mask from a dataset.
        Images go through preprocessing before returning
        Args:
            i: type=int, Index element

        Returns:
        Augmented image and its mask
        """
        image = cv2.imread(self.images_fps[i])
        mask = cv2.imread(self.masks_fps[i])

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        """
        Method for getting count images
        Returns:
            type=int, count images
        """
        return len(self.ids)
