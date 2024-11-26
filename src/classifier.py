"""Module to load a pre-trained classifier and preprocessing."""

import timm
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.dataset import get_preprocessing


def get_transformer_classifier(model_name, device):
    """
    Get model for image classification pre-trained, of the types AutoImage.

    USAGE:
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    """
    processor = AutoImageProcessor.from_pretrained(
        model_name, do_convert_rgb=True, input_data_format=None
    )  # , do_rescale=True)

    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    model.to(device)

    return model, processor


def get_timm_classifier(model_name, dataset_name, device):
    """Get a pretrained classifer and prepare the adequaate preprocessing, with the timm library."""
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    model.to(device)

    preprocess = get_preprocessing(dataset_name)

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
