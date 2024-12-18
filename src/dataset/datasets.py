"""This module contains the classes for the datasets based on the base dataset."""

import torch
from transformers import AutoImageProcessor

from .aux import TRANSFORMATIONS
from .base import SyntheticDataset


class OtherDataset(SyntheticDataset):
    """Dataset that uses a manual preprocessing step."""

    def __init__(self, dataset_name, n_classes, images, device):
        """Construct the PretrainedTransformer class."""
        transform = TRANSFORMATIONS[dataset_name]
        super().__init__(dataset_name, n_classes, images, transform, device)

    def image_to_tensor(self):
        """Return the entire dataset as tensors."""
        return torch.stack([self.transform(img) for img in self.images])

    def sample_as_tensor(self, n):
        """Sample n random images and returns them as tensors."""
        sampled_images = self.sample(n)
        tensor_images = torch.stack([self.transform(img) for img in sampled_images])
        return tensor_images, sampled_images


class TransfomerDataset(SyntheticDataset):
    """Dataset that uses a pretrained transformer for preprocessing."""

    def __init__(self, dataset_name, n_classes, model_name, images, device):
        """Construct the PretrainedTransformer class."""
        transform = AutoImageProcessor.from_pretrained(model_name)
        super().__init__(dataset_name, n_classes, images, transform, device)

    def image_to_tensor(self):
        """Return the logits of the model for the given images."""
        # Check the type and size of the first image
        tensor_images = self.transform(images=self.images, return_tensors="pt")
        return tensor_images["pixel_values"]

    def sample_as_tensor(self, n):
        """Sample n random images and returns them as tensors."""
        sampled_images = self.sample(n)
        tensor_images = self.transform(images=sampled_images, return_tensors="pt")
        return tensor_images["pixel_values"], sampled_images
