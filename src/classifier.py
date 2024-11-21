"""Module to load a pre-trained classifier and preprocessing."""

# from torchvision.models import resnet50, ResNet50_Weights
import timm
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification


def get_vit_classifier(model_name, device):
    """_summary_ Get a VIT model for image classification pre-trained with CIFAR10.

    Returns:
        _type_: _description_ model and feature_extractor
    """
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    model.eval()
    model.to(device)

    # USAGE:
    # inputs = feature_extractor(images=image, return_tensors="pt")
    # outputs = model(**inputs)
    # preds = outputs.logits.argmax(dim=1)
    # classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # classes[preds[0]]

    return model, feature_extractor


def get_vit_classifierget_classifier(device):
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
