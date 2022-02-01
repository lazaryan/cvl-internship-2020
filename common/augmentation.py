"""
The script contains functions for augmentation and image preprocessing
"""

import albumentations as albu


def get_training_augmentation():
    """
    Function for getting augmentation for image
    Returns:
        array transforms methods
    """
    train_transform = [
        albu.Resize(height=416, width=576, p=1),
        albu.VerticalFlip(p=0.5),
        albu.OneOf([
            albu.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            albu.GridDistortion(p=0.5),
            albu.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
        ], p=0.8),
        albu.CLAHE(p=0.8),
        albu.RandomBrightnessContrast(p=0.8),
        albu.RandomGamma(p=0.8)
    ]

    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(416, 576)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    """
    Function for convert images to tensor
    Args:
        x: image

    Returns:
        new Tensor
    """
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
