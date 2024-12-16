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

    def image_as_tensors(self, images):
        """Return the entire dataset as tensors."""
        return torch.stack([self.transform(img) for img in images]).to(self.device)


class TransfomerDataset(SyntheticDataset):
    """Dataset that uses a pretrained transformer for preprocessing."""

    def __init__(self, dataset_name, n_classes, model_name, images, device):
        """Construct the PretrainedTransformer class."""
        transform = AutoImageProcessor.from_pretrained(model_name, do_convert_rgb=True, input_data_format=None)
        super().__init__(dataset_name, n_classes, images, transform, device)

    def image_as_tensors(self, images):
        """Return the logits of the model for the given images."""
        # TODO: check if this is why MNIST is not working properly
        # images = [img.convert("RGB") for img in images]
        tensor_images = self.transform(images=images, return_tensors="pt").to(self.device)
        return tensor_images["pixel_values"]
