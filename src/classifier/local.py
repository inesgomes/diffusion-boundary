"""Module for locally trained classifiers."""

import os

import torch

from src.classifier.base import BaseClassifier
from src.classifier.cnn import CNN
from src.dataset import get_preprocessing


class LocalClassifier(BaseClassifier):
    """Class for pre-trained models from the timm library."""

    def __init__(self, model_path, dataset, n_classes, device):
        """Construct the LocalClassifier class."""
        super().__init__(model_path, dataset, n_classes, device)

        # image preprocessor
        self.preprocessor = get_preprocessing(dataset)

        # model
        self.model = self.construct_classifier_from_checkpoint(model_path)
        self.model.to(device)
        self.model.eval()

    def predict(self, tensor_images):
        """Return the logits of the model for the given images."""
        return self.model(tensor_images)

    def pil_to_tensor(self, images):
        """Return the logits of the model for the given images."""
        tensor_images = torch.stack([self.preprocessor(img) for img in images])
        return tensor_images.to(self.device)

    def construct_classifier_from_checkpoint(self, path):
        """Code from GASTeN library, to load a model locally trained."""
        # get model params and state from checkpoint
        cp = torch.load(os.path.join(path, "classifier.pth"))
        model_params = cp["params"]

        # construct classifier (in this case, a CNN)
        model = CNN(model_params["img_size"], model_params["n_classes"], model_params["nf"])
        model.load_state_dict(cp["state"])
        return model
