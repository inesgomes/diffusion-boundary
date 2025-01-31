"""
Use this file to load and preprocess datasets.

Returns:
    _type_: _description_ dataset
"""

import random
from functools import partial

from datasets import Dataset, load_dataset
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


def get_tst_dataset(dataset_name, n_samples=None, subset=None):
    """Get a dataset. Needs to load the full dataset into memory (may be a drawback for large datasets)."""
    # get a dataset from huggingface
    samples = load_dataset(dataset_name, split="test", streaming=True)
    key = samples.column_names[0]  # (img or image depending on the dataset)
    # select only the relevant labels, if that is required
    if subset is not None:
        samples = samples.filter(lambda x: x["label"] in subset)
    # select number of samples
    if n_samples is not None:
        dataset_subset = samples.select(random.sample(range(len(samples)), n_samples))
        return dataset_subset[key], dataset_subset["label"]
    return samples[key], samples["label"]


def gen_from_iterable_dataset(iterable_ds):
    """Help generate a dataset from an iterable."""
    yield from iterable_ds


def get_tst_dataset_streaming(dataset_name, dataset_split="test", n_samples=None, subset=None):
    """Get a dataset in streaming mode. This is useful for large datasets as the dataset is not loaded into memory (only after sampling)."""
    if subset is not None:
        raise NotImplementedError("Subset is not yet implemented for streaming datasets")

    # get a dataset from huggingface in streaming mode
    dataset = load_dataset(dataset_name, split=dataset_split, streaming=True)
    dataset_sample = dataset.shuffle(seed=42, buffer_size=5000).take(n_samples)

    # from iterable to dataset
    ds = Dataset.from_generator(partial(gen_from_iterable_dataset, dataset_sample), features=dataset_sample.features)

    # the class labels for the dataset
    class_labels = ds.info.features["label"].names
    # (img or image depending on the dataset)
    key = ds.column_names[0]
    return ds[key], ds["label"], class_labels
