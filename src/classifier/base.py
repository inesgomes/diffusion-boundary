"""Base Classifier class for pre-trained models."""


# Base Classifier class
class BaseClassifier:
    """Base class for pre-trained models."""

    def __init__(self, model_name, dataset, n_classes, device):
        """Construct the Pretrained class."""
        self.model_name = model_name
        self.dataset = dataset
        self.n_classes = n_classes
        self.device = device

    def predict(self, tensor_images):
        """Run forward pass and return predictions."""
        raise NotImplementedError("Subclasses should implement this method")

    def pil_to_tensor(self, images):
        """Run forward pass and return predictions from PIL images."""
        raise NotImplementedError("Subclasses should implement this method")

    def predict_from_pil(self, images):
        """Return the logits of the model for the given images."""
        tensor_images = self.pil_to_tensor(images)
        return self.predict(tensor_images)

    def get_dataset_name(self):
        """Return the name of the dataset."""
        return self.dataset

    def get_n_classes(self):
        """Return the number of classes in the dataset."""
        return self.n_classes
