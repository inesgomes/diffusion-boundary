"""Module for the base class of the synthetic dataset."""

import random

from torch.utils.data import Dataset

from .aux import TRANSFORMATIONS


class SyntheticDataset(Dataset):
    """Class for custom datasets that includes the synthetic samples and the reference to the real ones."""

    def __init__(self, dataset_name, n_classes, images, transform):
        """Construct the SyntheticDataset class."""
        self.images = images
        self.transform = transform
        self.transform_norm = TRANSFORMATIONS["norm"]
        self.dataset_name = dataset_name
        self.n_classes = n_classes
        self.use_default_transformation = True
        self.use_convert_rgb = False

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.images)

    def sample(self, n):
        """Sample n random images."""
        return random.sample(self.images, min(n, len(self.images)))

    def sample_to_tensor(self, n):
        """Sample n random images and returns them as tensors."""
        raise NotImplementedError("Subclasses should implement this method")

    def set_convert_rgb(self, use_convert_rgb):
        """Set the convert_rgb flag."""
        self.use_convert_rgb = use_convert_rgb

    def set_default_transformation(self, use_default_transformation):
        """Set the default_transformation flag."""
        self.use_default_transformation = use_default_transformation

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
