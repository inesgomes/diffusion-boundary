"""Base Classifier class for pre-trained models."""

from torch.nn import Dropout


class BaseClassifier:
    """Base class for pre-trained models."""

    def __init__(self, model, device="cpu"):
        """Construct the Pretrained class."""
        self.model = model
        self.model.to(device)
        self.model.eval()

    def get_model(self):
        """Return the model."""
        return self.model

    def predict(self, tensor_images):
        """Run forward pass and return predictions."""
        raise NotImplementedError("Subclasses should implement this method")

    def set_dropout(self, dropout_p=0.1):
        """Manually set dropout probability in a ViT model."""
        for module in self.model.modules():
            if isinstance(module, Dropout):
                module.p = dropout_p

    def set_train(self):
        """Set model to training mode."""
        self.model.train()

    def set_eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
