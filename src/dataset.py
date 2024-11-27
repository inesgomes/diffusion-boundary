"""
Use this file to load and preprocess datasets.

Returns:
    _type_: _description_ dataset
"""

from datasets import load_dataset
from torchvision import transforms


def get_preprocessing(dataset_name):
    """Get preprocessing for a dataset, given its name. The normalization values are taken from documentation."""
    # Preprocessing for classifier input
    if dataset_name == "cifar10":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),  # Normalize CIFAR10 images with mean and std
            ]
        )
    if dataset_name == "mnist":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),  # Normalize grayscale images with mean and std for MNIST
            ]
        )

    raise ValueError(f"Dataset {dataset_name} not supported.")


def get_labels(dataset_name):
    """Get the labels for a dataset, given its name."""
    if dataset_name == "cifar10":
        return ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    if dataset_name == "mnist":
        return list(range(10))
    raise ValueError(f"Dataset {dataset_name} not supported.")


def get_dataset(dataset_name):
    """Get a dataset."""
    return load_dataset(dataset_name)
