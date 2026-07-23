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
import torchvision.transforms.functional as TF
from datasets import Dataset, get_dataset_config_info, load_dataset, load_from_disk


class ResizeCPUBackward(torch.autograd.Function):
    """Antialiased resize with the gradient computed on the CPU.

    ``upsample_{bilinear,bicubic}2d_aa_backward_cuda`` accumulates with atomics and has no
    deterministic CUDA implementation, and it is reached on every classifier-guidance step, so it
    makes generation non-reproducible. A resize is a linear map with no bias, so evaluating its
    vector-jacobian product on a zero input gives exactly the same gradient, without the atomics.
    The forward is untouched, so the images the classifier sees are unchanged.
    """

    @staticmethod
    def forward(ctx, img, size, interpolation):
        """Resize on the GPU, exactly as torchvision.transforms.Resize would."""
        ctx.args = (img.shape, img.dtype, img.device, size, interpolation)
        return TF.resize(img, size, interpolation=interpolation, antialias=True)

    @staticmethod
    def backward(ctx, grad_output):
        """Replay the resize on the CPU and let autograd transpose it there."""
        shape, dtype, device, size, interpolation = ctx.args
        probe = torch.zeros(shape, device="cpu", requires_grad=True)
        with torch.enable_grad():
            resized = TF.resize(probe, size, interpolation=interpolation, antialias=True)
        grad = torch.autograd.grad(resized, probe, grad_output.to("cpu", torch.float32))[0]
        return grad.to(device=device, dtype=dtype), None, None


def deterministic_resize(size, interpolation):
    """Drop-in for ``T.Resize(size, interpolation, antialias=True)`` with a deterministic backward."""
    return T.Lambda(lambda img: ResizeCPUBackward.apply(img, size, interpolation))


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
            deterministic_resize((224, 224), T.InterpolationMode.BILINEAR),
            T.Lambda(lambda x: x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x),
            T.Lambda(lambda x: (x + 1) / 2),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
    # manual imitation of the default transformation of the pretrained model - tensor option
    "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10_tensor": T.Compose(
        [
            deterministic_resize((224, 224), T.InterpolationMode.BILINEAR),
            T.Lambda(lambda x: (x + 1) / 2),  # Convert from [-1,1] -> [0,1]
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
    # manual imitation of the default transformation of the pretrained model microsoft/resnet-50 - tensor option
    # Mirrors ConvNextFeatureExtractor as declared in its preprocessor_config.json
    # (crop_pct=0.875, size=224, resample=3 -> BICUBIC): resize the shortest edge to
    # int(224 / 0.875) = 256, then center crop 224. Every op stays differentiable, so the
    # guidance gradient still flows from the classifier back to the latent.
    "microsoft/resnet-50_tensor": T.Compose(
        [
            deterministic_resize(256, T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    # Mirrors ViTImageProcessor as declared in its preprocessor_config.json (size=224,
    # image_mean/image_std=0.5, default resample=BILINEAR): squash to exactly 224x224, no crop.
    "google/vit-base-patch16-224_tensor": T.Compose(
        [
            deterministic_resize((224, 224), T.InterpolationMode.BILINEAR),
            T.ConvertImageDtype(torch.float32),
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

    path = os.getenv("FILESDIR") + f"/dataset/{dataset_name}_{dataset_split}_{n_samples}_{subset_name}"
    print("Reference dataset path: ", path)

    if os.path.exists(path):
        ds = load_from_disk(path)
    else:
        ds = get_tst_dataset_streaming(dataset_name, dataset_split, n_samples, subset_idx)
        print(f"saving reference {dataset_split} dataset...")
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
