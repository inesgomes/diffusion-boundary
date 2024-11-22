"""
Use this file to load and preprocess datasets.

Returns:
    _type_: _description_ dataset
"""

from datasets import load_dataset


def get_cifar10_dataset(dataset_name):
    """Get a dataset."""
    return load_dataset(dataset_name)
