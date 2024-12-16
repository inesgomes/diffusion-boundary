"""Module for the base class of the synthetic dataset."""

import random


class SyntheticDataset:
    """Class for custom datasets that includes the synthetic samples and the reference to the real ones."""

    def __init__(self, dataset_name, n_classes, images, transform, device="cpu"):
        """Construct the SyntheticDataset class."""
        self.images = images
        self.transform = transform
        self.device = device
        self.dataset_name = dataset_name
        self.n_classes = n_classes

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.images)

    def sample(self, n):
        """Sample n random images."""
        return random.sample(self.images, min(n, len(self.images)))

    def image_as_tensors(self, images):
        """Run forward pass and return predictions."""
        raise NotImplementedError("Subclasses should implement this method")

    def as_tensors(self):
        """Return the entire dataset as tensors."""
        return self.image_as_tensors(self.images)

    def sample_as_tensors(self, n):
        """Sample n random images and returns them as tensors."""
        sampled_images = self.sample(n)
        return self.image_as_tensors(sampled_images), sampled_images

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
