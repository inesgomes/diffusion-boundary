"""Module for the PretrainedTransformer class."""

from torch.nn import functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.classifier.pretrained_base import Pretrained


class PretrainedTransformer(Pretrained):
    """Class for pre-trained models from the transformers library."""

    def __init__(self, model_name, dataset, device):
        """Construct the PretrainedTransformer class."""
        super().__init__(model_name, dataset, device)

        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.preprocessor = AutoImageProcessor.from_pretrained(model_name, do_convert_rgb=True, input_data_format=None)

    def predict(self, tensor_images):
        """Return the logits of the model for the given images."""
        logits = self.model(tensor_images).logits
        return F.softmax(logits, dim=1)

    def pil_to_tensor(self, images):
        """Return the logits of the model for the given images."""
        images = [img.convert("RGB") for img in images]
        tensor_images = self.preprocessor(images=images, return_tensors="pt").to(self.device)
        return tensor_images["pixel_values"]
