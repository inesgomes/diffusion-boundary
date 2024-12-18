"""Module for the base class of the synthetic dataset."""

import random

import torch
from torchvision import transforms


class SyntheticDataset:
    """Class for custom datasets that includes the synthetic samples and the reference to the real ones."""

    def __init__(self, dataset_name, n_classes, images, transform, device="cpu"):
        """Construct the SyntheticDataset class."""
        self.images = images
        self.transform = transform
        self.device = device
        self.dataset_name = dataset_name
        self.n_classes = n_classes
        self.tensors = self.image_to_tensor()

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.images)

    def sample(self, n):
        """Sample n random images."""
        return random.sample(self.images, min(n, len(self.images)))

    def image_to_tensor(self):
        """Transform the images to tensors."""
        raise NotImplementedError("Subclasses should implement this method")

    def image_to_norm_tensor(self):
        """Transform the images to normalized tensors."""
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        return torch.stack([transform(img) for img in self.images])

    def sample_as_tensor(self, n):
        """Sample n random images and returns them as tensors."""
        raise NotImplementedError("Subclasses should implement this method")

    def get_transform(self):
        """Return the transform."""
        return self.transform

    def get_images(self):
        """Return the images."""
        return self.images

    def get_dataset_name(self):
        """Return the dataset name."""
        return self.dataset_name

    def get_n_classes(self):
        """Return the number of classes."""
        return self.n_classes

    def get_tensors(self):
        """Return the tensors."""
        return self.tensors

    def get_device(self):
        """Return the device."""
        return self.device
