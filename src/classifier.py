"""Module to load a pre-trained classifier and preprocessing."""

from torchvision import models, transforms


def get_classifier(device):
    """Get a pretrained classifer (ResNet50) and prepare the adequaate preprocessing."""
    # Load a pre-trained classifier
    classifier = models.resnet50(pretrained=True)
    classifier.eval()
    classifier.to(device)

    # Preprocessing for classifier input
    preprocess = transforms.Compose(
        [
            #    transforms.Resize(256),
            #    transforms.CenterCrop(224),
            #    transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return classifier, preprocess
