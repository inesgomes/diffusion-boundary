"""Module for the base class of the synthetic dataset."""

import random
from typing import Literal

from torch.utils.data import Dataset

_TRANFORMATION = Literal["DEFAULT", "NORM", "NONE"]


class SyntheticDataset(Dataset):
    """Class for custom datasets that includes the synthetic samples and the reference to the real ones."""

    def __init__(self, dataset_name, n_classes, class_labels, images, transform, transform_t=None, transform_norm=None):
        """Construct the SyntheticDataset class."""
        self.images = images
        self.labels = None  # placeholder for labels
        self.transform = transform
        self.transform_t = transform_t
        self.transform_norm = transform_norm
        self.dataset_name = dataset_name
        self.n_classes = n_classes
        self.class_labels = class_labels
        self.use_transformation = "DEFAULT"
        self.use_convert_rgb = True  # dataset_name == "mnist"

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.images)

    def sample(self, n):
        """Sample n random images."""
        return random.sample(self.images, min(n, len(self.images)))

    def set_convert_rgb(self, use_convert_rgb: bool):
        """Set the convert_rgb flag."""
        self.use_convert_rgb = use_convert_rgb

    def set_use_transformation(self, transformation_type: _TRANFORMATION):
        """Set the default_transformation flag. Options: default, norm or none."""
        self.use_transformation = transformation_type

    def get_transform(self):
        """Return the transform."""
        return self.transform

    def get_images(self):
        """Return the images."""
        return self.images

    def get_labels(self):
        """Return the labels."""
        return self.labels

    def get_dataset_name(self):
        """Return the dataset name."""
        return self.dataset_name

    def get_n_classes(self):
        """Return the number of classes."""
        return self.n_classes

    def get_class_labels(self):
        """Return the class labels, ordered."""
        return self.class_labels

    def transform_images(self, images):
        """Transform images outside of the dataset, with the same transformation."""
        raise NotImplementedError("Subclasses should implement this method")

    def set_images(self, images):
        """Set the images."""
        self.images = images

    def set_labels(self, labels):
        """Set the labels."""
        self.labels = labels

    def get_class_idx(self, label):
        """Get the index of the class, given its label."""
        return self.class_labels.index(label) if label in self.class_labels else None
