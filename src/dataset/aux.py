"""
Use this file to load and preprocess datasets.

Returns:
    _type_: _description_ dataset
"""

import random

from datasets import load_dataset
from torchvision import transforms

TRANSFORMATIONS = {
    "cifar10": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.49139968, 0.48215827, 0.44653124),
                std=(0.24703233, 0.24348505, 0.26158768),
            ),
        ]
    ),
    "cifar10_norm": transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    ),
    "mnist": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    ),
}

LABELS = {
    "cifar10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    "mnist": list(range(10)),
}


def get_tst_dataset(dataset_name, subset, n_samples):
    """Get a dataset. Needs to load the full dataset into memory (may be a drawback for large datasets)."""
    # get a dataset from huggingface
    samples = load_dataset(dataset_name, split="test")
    # select only the relevant labels, if that is required
    if subset is not None:
        samples = samples.filter(lambda x: x["label"] in subset)
    # select number of samples
    dataset_subset = samples.select(random.sample(range(len(samples)), n_samples))
    # img or image depending on the dataset
    key = dataset_subset.column_names[0]
    # pil images and labels
    return dataset_subset[key], dataset_subset["label"]
