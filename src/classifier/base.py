"""Base Classifier class for pre-trained models."""


# Base Classifier class
class Pretrained:
    """Base class for pre-trained models."""

    def __init__(self, model_name, dataset, device):
        """Construct the Pretrained class."""
        self.model_name = model_name
        self.dataset = dataset
        self.device = device

    def predict(self, tensor_images):
        """Run forward pass and return predictions."""
        raise NotImplementedError("Subclasses should implement this method")

    def pil_to_tensor(self, images):
        """Run forward pass and return predictions from PIL images."""
        raise NotImplementedError("Subclasses should implement this method")

    def get_dataset_name(self):
        """Return the name of the dataset."""
        return self.dataset
