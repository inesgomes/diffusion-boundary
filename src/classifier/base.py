"""Base Classifier class for pre-trained models."""

import torch
import torch.nn.functional as F
from torch.nn import Dropout, Module, Parameter
from tqdm import tqdm


def extract_logits(output):
    """Return the logits tensor from a model output, whatever library the model comes from.

    Transformers models wrap the logits in an output object, while timm and locally trained
    models return the logits tensor directly.
    """
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError(
        f"Unsupported model output type '{type(output).__name__}': expected a torch.Tensor "
        "or an object exposing a .logits attribute."
    )


class TemperatureScaler(Module):
    """A temperature scaler module for calibrating model logits."""

    def __init__(self):
        """Construct the TemperatureScaler class."""
        super().__init__()
        self.temperature = Parameter(torch.ones(1))

    def forward(self, logits):
        """Scale the logits by the temperature."""
        return logits / self.temperature.expand(logits.size(0), 1)


class BaseClassifier:
    """Base class for pre-trained models."""

    def __init__(self, model, device):
        """Construct the Pretrained class."""
        self.device = device
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.scaler = None  # calibration scaler, starting as None only instantiated if needed

    def get_model(self):
        """Return the model."""
        return self.model

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

    def predict(self, tensor_images):
        """Run forward pass and return predictions."""
        raise NotImplementedError("Subclasses should implement this method")

    def raw_logits(self, logits):
        """Undo the temperature scaling ``predict`` applies, recovering the model's own logits.

        Scaling is a division by a positive scalar, so multiplying back is exact. Metrics that
        must not depend on the calibration fit take these instead of the logits ``predict``
        returns. The temperature is detached: it is fixed after calibration, and any gradient
        should reach the logits, not it.
        """
        if self.scaler is None:
            return logits
        return logits * self.scaler.temperature.detach().to(logits.device)

    def _train_temperature_scaler(self, dataloader):
        """Train a temperature scaler using the provided data loader."""
        self.scaler = TemperatureScaler().to(self.device)
        self.scaler.train()

        optimizer = torch.optim.LBFGS([self.scaler.temperature], lr=0.01, max_iter=50)

        all_logits = []
        all_labels = []

        # Collect logits and labels for efficiency
        with torch.no_grad():
            for batch_images, batch_labels in tqdm(dataloader, desc="Optimizing temperature scaler"):
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                logits = extract_logits(self.model(batch_images))
                all_logits.append(logits)
                all_labels.append(batch_labels)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        def _closure():
            optimizer.zero_grad()
            scaled_logits = self.scaler(all_logits)
            loss = F.cross_entropy(scaled_logits, all_labels)
            loss.backward()
            return loss

        optimizer.step(_closure)
        print("Optimal temperature:", self.scaler.temperature.item())

    def calibrate(self, dataloader):
        """Calibrate the model using temperature scaling. Only used if no scaler is provided."""
        if self.scaler is None:
            self._train_temperature_scaler(dataloader)
