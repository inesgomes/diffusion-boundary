"""Module for locally trained classifiers."""

import os

import timm
import torch
from torch.nn import functional as F
from transformers import AutoModelForImageClassification

from .base import BaseClassifier
from .cnn import CNN


class LocalClassifier(BaseClassifier):
    """Class for pre-trained models from the timm library."""

    def __init__(self, model_path, device):
        """Construct the LocalClassifier class."""
        model = self.construct_classifier_from_checkpoint(model_path)
        super().__init__(model, device)

    def predict(self, tensor_images):
        """Return the logits of the model for the given images."""
        return self.model(tensor_images)

    def construct_classifier_from_checkpoint(self, path):
        """Code from GASTeN library, to load a model locally trained."""
        # get model params and state from checkpoint
        cp = torch.load(os.path.join(path, "classifier.pth"))
        model_params = cp["params"]

        # construct classifier (in this case, a CNN)
        model = CNN(model_params["img_size"], model_params["n_classes"], model_params["nf"])
        model.load_state_dict(cp["state"])
        return model


class PretrainedOther(BaseClassifier):
    """Class for pre-trained models from the timm library."""

    def __init__(self, model_name, device):
        """Construct the PretrainedOther class."""
        model = timm.create_model(model_name, pretrained=True)
        super().__init__(model, device)

    def predict(self, tensor_images):
        """Return the logits of the model for the given images."""
        logits = self.model(tensor_images)
        probs = F.softmax(logits, dim=1)
        return probs


class PretrainedTransformer(BaseClassifier):
    """Class for pre-trained models from the transformers library."""

    def __init__(self, model_name, device):
        """Construct the PretrainedTransformer class."""
        model = AutoModelForImageClassification.from_pretrained(model_name)
        super().__init__(model, device)

    def predict(self, tensor_images):
        """Return the logits of the model for the given images."""
        logits = self.model(tensor_images).logits
        probs = F.softmax(logits, dim=1)
        return probs
