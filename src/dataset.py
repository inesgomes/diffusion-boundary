"""
Use this file to load and preprocess datasets.

Returns:
    _type_: _description_ dataset
"""

from datasets import load_dataset


def get_cifar10_dataset():
    """Get the CIFAR10 dataset. It is 32x32 pixels."""
    return load_dataset("uoft-cs/cifar10")
