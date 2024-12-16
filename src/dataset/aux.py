"""
Use this file to load and preprocess datasets.

Returns:
    _type_: _description_ dataset
"""

from itertools import islice

from datasets import load_dataset
from torchvision import transforms

TRANSFORMATIONS = {
    "cifar10": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                # mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) # CIFAR10 mean and std
            ),
        ]
    ),
    "mnist": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),  # MNIST mean and std
        ]
    ),
}

LABELS = {
    "cifar10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    "mnist": list(range(10)),
}


def get_tst_dataset(dataset_name, subset, n_samples):
    """Get a dataset."""
    # get a dataset from huggingface
    samples = load_dataset(dataset_name, split="test", streaming=True)
    # select only the relevant labels, if that is required
    if subset is not None:
        samples = samples.filter(lambda example: example["label"] in subset)
    # select number of samples
    # TODO fix this error -> we are not giving images as PIL (i think) ?
    return list(islice(samples, n_samples))
    # to right format
    # return [Image.open(path) for path in selected_samples]
