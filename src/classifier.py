"""Module to load a pre-trained classifier and preprocessing."""

# from torchvision.models import resnet50, ResNet50_Weights
import timm
from torchvision import transforms


def get_classifier(device):
    """Get a pretrained classifer (ResNet50) and prepare the adequaate preprocessing."""
    # Load a pre-trained classifier
    # standard pre-trained model
    # classifier = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # resnet10 trained on cifar10
    model = timm.create_model("hf_hub:edadaltocg/resnet50_cifar10", pretrained=True)
    model.eval()
    model.to(device)

    # Preprocessing for classifier input
    preprocess = transforms.Compose(
        [
            #    transforms.Resize(256),
            #    transforms.CenterCrop(224),
            #    transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, preprocess
