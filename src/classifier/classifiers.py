"""Module for locally trained classifiers."""

import os

import timm
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForImageClassification

from .base import BaseClassifier, extract_logits
from .cnn import CNN


class LocalClassifier(BaseClassifier):
    """Class for pre-trained models from the timm library."""

    def __init__(self, model_path, device="cpu"):
        """Construct the LocalClassifier class."""
        model = self.construct_classifier_from_checkpoint(model_path)
        super().__init__(model, device)

    def predict(self, tensor_images):
        """Return the logits of the model for the given images."""
        logits = self.model(tensor_images)
        if self.scaler is not None:
            logits = self.scaler(logits)
        return F.softmax(logits, dim=1), logits

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

    def __init__(self, model_name, device="cpu"):
        """Construct the PretrainedOther class."""
        model = timm.create_model(model_name, pretrained=True)
        super().__init__(model, device)

    def predict(self, tensor_images):
        """Return the logits of the model for the given images."""
        logits = self.model(tensor_images)
        if self.scaler is not None:
            logits = self.scaler(logits)
        probs = F.softmax(logits, dim=1)
        return probs, logits

    def soft_corrupt_classifier(self, std=0.05):
        """Softly corrupt the classifier weights by adding Gaussian noise."""
        with torch.no_grad():
            classifier = self.model.classifier
            if isinstance(classifier, nn.Sequential):
                for layer in classifier:
                    if isinstance(layer, nn.Linear):
                        layer.weight += torch.randn_like(layer.weight) * std
                        layer.bias += torch.randn_like(layer.bias) * std
                        break
            elif isinstance(classifier, nn.Linear):
                classifier.weight += torch.randn_like(classifier.weight) * std
                classifier.bias += torch.randn_like(classifier.bias) * std
            else:
                raise TypeError(f"Unsupported classifier type: {type(classifier)}")


class PretrainedTransformer(BaseClassifier):
    """Class for pre-trained models from the transformers library."""

    def __init__(self, model_name, device="cpu"):
        """Construct the PretrainedTransformer class."""
        model = AutoModelForImageClassification.from_pretrained(model_name, cache_dir=os.getenv("HF_MODELS_CACHE"))
        super().__init__(model, device)

    def predict(self, tensor_images):
        """Return the logits of the model for the given images."""
        logits = extract_logits(self.model(tensor_images))
        if self.scaler is not None:
            logits = self.scaler(logits)
        probs = F.softmax(logits, dim=1)
        return probs, logits

    def soft_corrupt_classifier(self, std=0.05):
        """Softly corrupt the classifier weights by adding Gaussian noise."""
        with torch.no_grad():
            classifier = self.model.classifier
            if isinstance(classifier, nn.Sequential):
                for layer in classifier:
                    if isinstance(layer, nn.Linear):
                        layer.weight += torch.randn_like(layer.weight) * std
                        layer.bias += torch.randn_like(layer.bias) * std
                        break
            elif isinstance(classifier, nn.Linear):
                classifier.weight += torch.randn_like(classifier.weight) * std
                classifier.bias += torch.randn_like(classifier.bias) * std
            else:
                raise TypeError(f"Unsupported classifier type: {type(classifier)}")
