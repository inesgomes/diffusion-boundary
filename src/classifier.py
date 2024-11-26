"""Module to load a pre-trained classifier and preprocessing."""

import sys

import timm
import torch
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from transformers import AutoImageProcessor, AutoModelForImageClassification


def get_classifier(lib_name, model_name, device):
    """Get a pre-trained classifier model and preprocessing according to library."""
    if lib_name == "transformers":
        return get_transformer_classifier(model_name, device)
    if lib_name == "timm":
        return get_timm_classifier(model_name, device)
    print(f"Library {lib_name} not implemented.")
    sys.exit(1)


def get_transformer_classifier(model_name, device):
    """Get model for image classification pre-trained, of the types AutoImage."""
    processor = AutoImageProcessor.from_pretrained(model_name)
    processor.do_rescale = False

    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    model.to(device)

    # USAGE:
    # inputs = feature_extractor(images=image, return_tensors="pt")
    # outputs = model(**inputs)
    return model, processor


def get_timm_classifier(model_name, device):
    """Get a pretrained classifer and prepare the adequaate preprocessing, with the timm library."""
    # resnet10 trained on cifar10
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    model.to(device)

    # Preprocessing for classifier input
    preprocess = transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, preprocess


def finetune_classifier():
    """
    Finetune a ResNet50 for our specific task.

    # TODO finish this method
    """
    # Preprocessing for classifier input
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # Load a pre-trained classifier
    classifier = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # TODO finetune

    return classifier, preprocess


def get_classifier_cifar10(device):
    """Temporary method to load a pre-trained model from huggingface, due to an issue with timm."""
    model = timm.create_model("resnet18", pretrained=False)

    # override model
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = torch.nn.Identity()  # type: ignore
    model.fc = torch.nn.Linear(512, 10)
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin",
            map_location=device,
            file_name="resnet18_cifar10.pth",
        )
    )
    model.eval()
    model.to(device)

    # Preprocessing for classifier input
    preprocess = transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, preprocess
