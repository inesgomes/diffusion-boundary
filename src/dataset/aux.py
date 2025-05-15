"""
Use this file to load and preprocess datasets.

Returns:
    _type_: _description_ dataset
"""

import os
from collections import defaultdict
from functools import partial

import torch
import torchvision.transforms as T
from datasets import Dataset, get_dataset_config_info, load_dataset, load_from_disk

TRANSFORMATIONS = {
    "mnist": T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    ),
    # manual imitation of the default transformation of the pretrained model - tensor option
    "farleyknight-org-username/vit-base-mnist_tensor": T.Compose(
        [
            T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
            T.Lambda(lambda x: x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x),
            T.Lambda(lambda x: (x + 1) / 2),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
    # manual imitation of the default transformation of the pretrained model - tensor option
    "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10_tensor": T.Compose(
        [
            T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
            T.Lambda(lambda x: (x + 1) / 2),  # Convert from [-1,1] -> [0,1]
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
    # manual imitation of the default transformation of the pretrained model microsoft/resnet-50 - tensor option
    "microsoft/resnet-50_tensor": T.Compose(
        [
            T.Resize(224, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(int(224 * 0.875)),  # to comment
            T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),  # to comment
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    # normalized between [-1, 1] option
    "microsoft/resnet-50_norm": T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize(224, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(int(224 * 0.875)),
            T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
}


def gen_from_iterable_dataset(iterable_ds):
    """Help generate a dataset from an iterable."""
    yield from iterable_ds


def gen_balanced_samples(iterable_ds, subset, n_samples):
    """Help generate a dataset from an iterable, filtering by class but only the number of samples asked."""
    class_counts = defaultdict(int)
    samples_per_class = n_samples / len(subset)

    for example in iterable_ds:
        label = example["label"]
        if label in subset and class_counts[label] < samples_per_class:
            yield example
            class_counts[label] += 1
            if sum(class_counts.values()) >= n_samples:
                break


def get_tst_dataset(dataset_name, dataset_split="test", n_samples=1, subset=None):
    """Given a dataset name, gets its labels and check if we have the dataset in disk or if we should get it in streaming mode."""
    # Get label names from configuration
    config_info = get_dataset_config_info(dataset_name, "default")
    class_labels = config_info.features["label"].names

    # get subset index given the labels
    subset_idx = [class_labels.index(name) for name in subset] if subset else []

    # check if dataset is in disk
    subset_name = "_".join(map(str, subset_idx))

    path = os.getenv("FILESDIR") + f"/dataset/{dataset_name}_{n_samples}_{subset_name}"
    print("Reference dataset path: ", path)

    if os.path.exists(path):
        ds = load_from_disk(path)
    else:
        ds = get_tst_dataset_streaming(dataset_name, dataset_split, n_samples, subset_idx)
        print("saving reference dataset...")
        os.makedirs(path, exist_ok=True)
        ds.save_to_disk(path)

    key = ds.column_names[0]  # (img or image depending on the dataset)
    return ds[key], ds["label"], class_labels


def get_tst_dataset_streaming(dataset_name, dataset_split, n_samples, subset_idx):
    """Get a dataset in streaming mode. This is useful for large datasets as the dataset is not loaded into memory (only after sampling)."""
    # get a dataset from huggingface in streaming mode
    dataset = load_dataset(dataset_name, split=dataset_split, streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=10000)

    # from iterable to dataset
    if len(subset_idx) > 0:
        generator = partial(gen_balanced_samples, dataset, subset_idx, n_samples)
    else:
        dataset_sample = dataset.take(n_samples)
        generator = partial(gen_from_iterable_dataset, dataset_sample)

    ds = Dataset.from_generator(generator, features=dataset.features)
    return ds
