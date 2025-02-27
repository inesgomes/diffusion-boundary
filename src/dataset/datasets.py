"""This module contains the classes for the datasets based on the base dataset."""

import torch
from PIL import Image
from transformers import AutoImageProcessor

from .aux import TRANSFORMATIONS
from .base import SyntheticDataset


class OtherDataset(SyntheticDataset):
    """Dataset that uses a manual preprocessing step."""

    def __init__(self, dataset_name, n_classes, class_labels, images):
        """Construct the PretrainedTransformer class."""
        transform = TRANSFORMATIONS[dataset_name]
        transform_norm = TRANSFORMATIONS[f"{dataset_name}_norm"]
        if f"{dataset_name}_norm" not in TRANSFORMATIONS:
            transform_norm = transform
        super().__init__(
            dataset_name, n_classes, class_labels, images, transform=transform, transform_norm=transform_norm
        )

    def __getitem__(self, idx):
        """Get an item from the dataset."""
        # correct transformation
        if self.transform and (self.use_transformation == "DEFAULT"):
            return self.transform(self.images[idx].convert("RGB") if self.use_convert_rgb else self.images[idx])
        # 0.5 normalization because of FID calculation
        if self.transform_norm and (self.use_transformation == "NORM"):
            return self.transform_norm(self.images[idx].convert("RGB") if self.use_convert_rgb else self.images[idx])
        return self.images[idx]

    def transform_images(self, images):
        """Transform images outside of the dataset, with the same transformation."""
        return torch.stack([self.transform(img) for img in images])


class TransfomerDataset(SyntheticDataset):
    """Dataset that uses a pretrained transformer for preprocessing."""

    def __init__(self, dataset_name, n_classes, class_labels, model_name, images):
        """Construct the PretrainedTransformer class."""
        transform = AutoImageProcessor.from_pretrained(model_name)
        # raise error if the transformation is not defined
        if f"{model_name}_tensor" not in TRANSFORMATIONS:
            raise ValueError(f"Tensor transformation for {model_name} not defined.")
        transform_tensor = TRANSFORMATIONS[f"{model_name}_tensor"]
        # normalized transformation, if exists
        transform_norm = TRANSFORMATIONS[f"{model_name}_norm"] if f"{model_name}_norm" in TRANSFORMATIONS else None

        super().__init__(
            dataset_name,
            n_classes,
            class_labels,
            images,
            transform=transform,
            transform_t=transform_tensor,
            transform_norm=transform_norm,
        )

    def __getitem__(self, idx):
        """Get an item from the dataset."""
        if self.use_transformation == "DEFAULT":
            if isinstance(self.images[idx], Image.Image) and self.transform:
                image = self.images[idx].convert("RGB") if self.use_convert_rgb else self.images[idx]
                return self.transform(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
            if isinstance(self.images[idx], torch.Tensor) and self.transform_t:
                return self.transform_t(self.images[idx])
        if self.use_transformation == "NORM":
            if self.transform_norm:
                return self.transform_norm(self.images[idx])
            if self.transform:
                image = self.images[idx].convert("RGB") if self.use_convert_rgb else self.images[idx]
                return self.transform(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        return self.images[idx]

    def transform_images(self, images):
        """Transform images outside of the dataset, with the same transformation."""
        if isinstance(images, Image.Image) and self.transform:
            return self.transform(images=images, return_tensors="pt")["pixel_values"].squeeze(1)
        if isinstance(images, torch.Tensor) and self.transform_t:
            if isinstance(images, list):
                images = torch.stack([self.transform(img) for img in images])
            return self.transform_t(images)
        raise ValueError("No transformation defined for the images.")
