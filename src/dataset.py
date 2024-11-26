"""
Use this file to load and preprocess datasets.

Returns:
    _type_: _description_ dataset
"""

from datasets import load_dataset

CIFAR_LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def get_labels(dataset_name):
    """Get the labels for a dataset, given its name."""
    if dataset_name == "cifar10":
        return CIFAR_LABELS
    if dataset_name == "mnist":
        return list(range(10))
    raise ValueError(f"Dataset {dataset_name} not supported.")


def get_cifar10_dataset(dataset_name):
    """Get a dataset."""
    return load_dataset(dataset_name)
