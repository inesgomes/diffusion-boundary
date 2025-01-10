"""This module contains the classes for the datasets based on the base dataset."""

import torch
from transformers import AutoImageProcessor

from .aux import TRANSFORMATIONS
from .base import SyntheticDataset


class OtherDataset(SyntheticDataset):
    """Dataset that uses a manual preprocessing step."""

    def __init__(self, dataset_name, n_classes, images):
        """Construct the PretrainedTransformer class."""
        transform = TRANSFORMATIONS[dataset_name]
        super().__init__(dataset_name, n_classes, images, transform)

    def __getitem__(self, idx):
        """Get an item from the dataset."""
        # correct transformation
        if self.transform and self.use_default_transformation:
            return self.transform(self.images[idx].convert("RGB") if self.use_convert_rgb else self.images[idx])
        # 0.5 normalization because of FID calculation
        if self.transform_norm:
            return self.transform_norm(self.images[idx].convert("RGB") if self.use_convert_rgb else self.images[idx])
        return self.images[idx]

    def sample_to_tensor(self, n):
        """Sample n random images and returns them as tensors."""
        sampled_images = self.sample(n)
        tensor_images = torch.stack(
            [self.transform(img.convert("RGB") if self.use_convert_rgb else img) for img in sampled_images]
        )
        return tensor_images, sampled_images


class TransfomerDataset(SyntheticDataset):
    """Dataset that uses a pretrained transformer for preprocessing."""

    def __init__(self, dataset_name, n_classes, model_name, images):
        """Construct the PretrainedTransformer class."""
        transform = AutoImageProcessor.from_pretrained(model_name)
        super().__init__(dataset_name, n_classes, images, transform)

    def __getitem__(self, idx):
        """Get an item from the dataset."""
        # correct transformation
        if self.transform:
            image = self.images[idx].convert("RGB") if self.use_convert_rgb else self.images[idx]
            return self.transform(images=image, return_tensors="pt")["pixel_values"]
        # 0.5 normalization because of FID calculation
        if self.transform_norm:
            return self.transform_norm(self.images[idx].convert("RGB") if self.use_convert_rgb else self.images[idx])
        return self.images[idx]

    def sample_to_tensor(self, n):
        """Sample n random images and returns them as tensors."""
        sampled_images = self.sample(n)
        images = [img.convert("RGB") for img in sampled_images] if self.use_convert_rgb else sampled_images
        tensor_images = self.transform(images=images, return_tensors="pt")
        return tensor_images["pixel_values"], sampled_images
