"""Base Classifier class for pre-trained models."""


# Base Classifier class
class BaseClassifier:
    """Base class for pre-trained models."""

    def __init__(self, model, device="cpu"):
        """Construct the Pretrained class."""
        self.model = model
        self.model.to(device)
        self.model.eval()

    def predict(self, tensor_images):
        """Run forward pass and return predictions."""
        raise NotImplementedError("Subclasses should implement this method")
