"""This module contains the classes for the datasets based on the base dataset."""

import torch
from transformers import AutoImageProcessor

from .aux import TRANSFORMATIONS
from .base import SyntheticDataset


class OtherDataset(SyntheticDataset):
    """Dataset that uses a manual preprocessing step."""

    def __init__(self, dataset_name, n_classes, class_labels, images):
        """Construct the PretrainedTransformer class."""
        transform = TRANSFORMATIONS[dataset_name]
        transform_norm = TRANSFORMATIONS[f"{dataset_name}_norm"]
        super().__init__(dataset_name, n_classes, class_labels, images, transform, transform_norm)

    def __getitem__(self, idx):
        """Get an item from the dataset."""
        # correct transformation
        if self.transform and (self.use_transformation == "default"):
            return self.transform(self.images[idx].convert("RGB") if self.use_convert_rgb else self.images[idx])
        # 0.5 normalization because of FID calculation
        if self.transform_norm and (self.use_transformation == "norm"):
            return self.transform_norm(self.images[idx].convert("RGB") if self.use_convert_rgb else self.images[idx])
        return self.images[idx]

    def transform_images(self, images):
        """Transform images outside of the dataset, with the same transformation."""
        return torch.stack([self.transform(img) for img in images])

    def sample_to_tensor(self, n):
        """Sample n random images and returns them as tensors."""
        sampled_images = self.sample(n)
        tensor_images = torch.stack(
            [self.transform(img.convert("RGB") if self.use_convert_rgb else img) for img in sampled_images]
        )
        return tensor_images, sampled_images


class TransfomerDataset(SyntheticDataset):
    """Dataset that uses a pretrained transformer for preprocessing."""

    def __init__(self, dataset_name, n_classes, class_labels, model_name, images):
        """Construct the PretrainedTransformer class."""
        transform = AutoImageProcessor.from_pretrained(model_name)
        super().__init__(dataset_name, n_classes, class_labels, images, transform)

    def __getitem__(self, idx):
        """Get an item from the dataset."""
        if self.transform:
            if self.use_transformation == "default":
                image = self.images[idx].convert("RGB") if self.use_convert_rgb else self.images[idx]
                return self.transform(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
            if self.use_transformation == "norm":
                # TODO not working
                image = self.images[idx].convert("RGB") if self.use_convert_rgb else self.images[idx]
                image_tensor = self.transform(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
                return (image_tensor * 2) - 1
        return self.images[idx]

    def transform_images(self, images):
        """Transform images outside of the dataset, with the same transformation."""
        images = [img.convert("RGB") if self.use_convert_rgb else img for img in images]
        return self.transform(images=images, return_tensors="pt")["pixel_values"].squeeze(1)

    def sample_to_tensor(self, n):
        """Sample n random images and returns them as tensors."""
        sampled_images = self.sample(n)
        images = [img.convert("RGB") for img in sampled_images] if self.use_convert_rgb else sampled_images
        tensor_images = self.transform(images=images, return_tensors="pt")
        return tensor_images["pixel_values"], sampled_images
