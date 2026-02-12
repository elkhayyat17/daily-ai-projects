"""
Data Augmentation & Transform Pipelines
Defines training and inference transforms for CIFAR-10 with ResNet50.
"""

import sys
from pathlib import Path
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def get_train_transforms() -> transforms.Compose:
    """
    Training transforms with data augmentation.

    Pipeline:
        1. Resize 32x32 → 224x224 (ResNet input size)
        2. Random horizontal flip
        3. Random color jitter (brightness, contrast)
        4. Random affine (slight rotation & translation)
        5. Normalize with ImageNet stats
        6. Random erasing for regularization
    """
    return transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=config.COLOR_JITTER_BRIGHTNESS,
                contrast=config.COLOR_JITTER_CONTRAST,
            )
        ], p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        transforms.RandomErasing(p=config.RANDOM_ERASING_PROB),
    ])


def get_val_transforms() -> transforms.Compose:
    """
    Validation/test transforms — no augmentation.
    Only resize and normalize.
    """
    return transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_inference_transforms() -> transforms.Compose:
    """
    Inference transforms for production.
    Handles arbitrary image sizes and formats.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def denormalize(tensor):
    """
    Reverse ImageNet normalization for visualization.

    Args:
        tensor: Normalized image tensor [C, H, W]

    Returns:
        Denormalized tensor clipped to [0, 1]
    """
    mean = config.IMAGENET_MEAN
    std = config.IMAGENET_STD
    denorm = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std],
    )
    return denorm(tensor).clamp(0, 1)
