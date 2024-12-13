"""Module for pre-trained models from the timm library."""

import timm
import torch
from torch.nn import functional as F

from src.classifier.base import BaseClassifier
from src.dataset import get_preprocessing


class PretrainedOther(BaseClassifier):
    """Class for pre-trained models from the timm library."""

    def __init__(self, model_name, dataset, n_classes, device):
        """Construct the PretrainedOther class."""
        super().__init__(model_name, dataset, n_classes, device)

        self.model = timm.create_model(model_name, pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        self.preprocessor = get_preprocessing(dataset)

    def predict(self, tensor_images):
        """Return the logits of the model for the given images."""
        logits = self.model(tensor_images)
        probs = F.softmax(logits, dim=1)
        return probs

    def pil_to_tensor(self, images):
        """Return the logits of the model for the given images."""
        tensor_images = torch.stack([self.preprocessor(img) for img in images])
        tensor_images = tensor_images.to(self.device)
        return tensor_images
