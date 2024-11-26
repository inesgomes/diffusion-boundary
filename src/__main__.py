"""This is the main file for the diffusion-boundary package."""

import argparse
import json
import os

import torch
import wandb
from dotenv import load_dotenv
from torch.nn import functional as F

from src.classifier import get_timm_classifier, get_transformer_classifier
from src.dataset import get_labels
from src.diffusion import get_custom_pipe, get_default_pipe
from src.utils import generate_run_id, load_configurations


def get_classifier(lib_name, model_name, dataset_name, device):
    """Get a pre-trained classifier model and preprocessing according to library."""
    if lib_name == "transformers":
        return get_transformer_classifier(model_name, device)
    if lib_name == "timm":
        return get_timm_classifier(model_name, dataset_name, device)
    raise ValueError(f"Library {lib_name} not implemented.")


def get_pipeline(pipeline, diff_type, model, device):
    """Get the pipeline for the diffusion model."""
    if pipeline == "default":
        return get_default_pipe(diff_type, model, device)
    return get_custom_pipe(diff_type, model, pipeline, device)


def get_arguments(pipeline_name, classifier, preprocessing, diffusion_settings):
    """Get arguments for the diffusion pipeline. Currently only for guidance pipeline."""
    if pipeline_name == "guidance":
        return {
            "classifier": classifier,
            "preprocessing": preprocessing,
            "alpha": diffusion_settings["args"]["alpha"],
        }
    return {}


def evaluate_classifier(classifier, lib, preprocess, dataset_name, images):
    """
    Evaluate the classifier on the generated images.

    # TODO: add this method to the relevant module
    """
    if lib == "transformers":
        images = [img.convert("RGB") for img in images]
        tensor_images = preprocess(images=images, return_tensors="pt")
        logits = classifier(tensor_images["pixel_values"]).logits
    else:
        tensor_images = torch.stack([preprocess(img) for img in images])
        with torch.no_grad():
            logits = classifier(tensor_images)

    probabilities = F.softmax(logits, dim=1)
    top_probs, top_indices = torch.topk(probabilities, k=10, dim=1)
    labels = get_labels(dataset_name)

    return {
        i: {labels[int(idx)]: round(prob.item(), 2) for idx, prob in zip(top_indices[i], top_probs[i])}
        for i in range(top_indices.size(0))
    }


def main(configuration):
    """Generate a sample image."""
    diffusion_settings = configuration["diffusion"]
    device = configuration["device"]
    group_name = configuration["dataset"]["name"] + "_" + configuration["diffusion"]["pipeline"]

    # init wandb
    wandb.init(
        project=configuration["project"],
        group=group_name,
        job_type=diffusion_settings["type"],
        entity=os.getenv("ENTITY"),
        name=generate_run_id(),
        config={
            "seed": configuration["seed"],
            "diffusion": diffusion_settings,
            "classsifier": configuration["classifier"]["name"],
        },
    )

    # get classifier specifications
    classifier = None
    preprocessing = None
    if configuration["classifier"] is not None:
        classifier, preprocessing = get_classifier(
            configuration["classifier"]["lib"],
            configuration["classifier"]["name"],
            configuration["dataset"]["name"],
            device,
        )
    # get diffusion pipeline
    pipe = get_pipeline(diffusion_settings["pipeline"], diffusion_settings["type"], diffusion_settings["name"], device)
    # get arguments for the pipeline
    args = get_arguments(diffusion_settings["pipeline"], classifier, preprocessing, diffusion_settings)
    # create generator
    generator = torch.Generator(device=device).manual_seed(configuration["seed"])

    # generate images
    images = pipe(
        generator=generator,
        num_inference_steps=diffusion_settings["args"]["num-inference-steps"],
        batch_size=diffusion_settings["args"]["batch-size"],
        **args,
    ).images
    # Log the grid as an image in WandB
    wandb.log({"sample_grid": [wandb.Image(img) for img in images]})

    # evaluate the synthetic images with the classifier (if available)
    if (classifier is not None) and (preprocessing is not None):
        results = evaluate_classifier(
            classifier, configuration["classifier"]["lib"], preprocessing, configuration["dataset"]["name"], images
        )
        print("RESULTS:", json.dumps(results, indent=4))

    # finish wandb
    wandb.finish()


if __name__ == "__main__":
    # TODO: method should be able to receive my trained classifier
    # load environment variables
    load_dotenv()

    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", required=True, help="Configuration file")
    my_args = parser.parse_args()
    # load information from config file
    config = load_configurations(my_args.config_path)

    # execute code
    main(config)
