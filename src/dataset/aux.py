"""
Use this file to load and preprocess datasets.

Returns:
    _type_: _description_ dataset
"""

import random
from functools import partial

import torch
import torchvision.transforms as T
from datasets import Dataset, load_dataset

TRANSFORMATIONS = {
    "mnist": T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    ),
    # manual imitation of the default transformation of the pretrained model, adapted for tensor
    "farleyknight-org-username/vit-base-mnist_tensor": T.Compose(
        [
            T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
            T.Lambda(lambda x: x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x),
            T.Lambda(lambda x: (x + 1) / 2),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
    # manual imitation of the default transformation of the pretrained model
    "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10_tensor": T.Compose(
        [
            T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
            T.Lambda(lambda x: (x + 1) / 2),  # Convert from [-1,1] -> [0,1]
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
    # manual imitation of the default transformation of the pretrained model microsoft/resnet-50
    "microsoft/resnet-50": T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize(224, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(int(224 * 0.875)),
            T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
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
            T.Lambda(lambda x: x * 2 - 1),
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    ),
    # tensor option
    "microsoft/resnet-50_tensor": T.Compose(
        [
            T.Resize(224, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(int(224 * 0.875)),
            T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
