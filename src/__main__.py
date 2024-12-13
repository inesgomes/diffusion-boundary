"""This is the main file for the diffusion-boundary package."""

import argparse
import json
import os
import random

import torch
import wandb
from diffusers import DDIMPipeline, DDPMPipeline, DiffusionPipeline, PNDMPipeline
from dotenv import load_dotenv

from src.classifier.local import LocalClassifier
from src.classifier.pretrained_other import PretrainedOther
from src.classifier.pretrained_transformer import PretrainedTransformer
from src.evaluation import (
    create_image_grid,
    create_probability_grid,
    label_synthetic_images,
)
from src.utils import generate_run_id, load_configurations


def create_classifier(lib_name, model_name, dataset_name, n_classes, device):
    """Create a pretrained model from a library."""
    if lib_name == "transformers":
        return PretrainedTransformer(model_name, dataset_name, n_classes, device)
    if lib_name == "timm":
        return PretrainedOther(model_name, dataset_name, n_classes, device)
    if lib_name == "local":
        return LocalClassifier(model_name, dataset_name, n_classes, device)
    return ValueError(f"Library {lib_name} not implemented.")


def create_pipeline(diff_type="ddpm", model="google/ddpm-cifar10-32", pipeline=None, device="cpu"):
    """
    General method to load pre-trained diffusion pipelines.

    Parameters:
        diff_type (str): The diffusion type ("ddpm", "ddim", "pndm", etc.).
        model (str): Pretrained model identifier.
        pipeline (str or None): Custom pipeline file path (if any).
        device (str): Device to load the pipeline on ("cpu" or "cuda").

    Returns:
        DiffusionPipeline: The loaded pipeline.
    """
    pipeline_classes = {
        "ddpm": DDPMPipeline,
        "ddim": DDIMPipeline,
        "pndm": PNDMPipeline,
    }

    # Select the pipeline class or fall back to the generic DiffusionPipeline
    pipeline_class = pipeline_classes.get(diff_type, DiffusionPipeline)

    # Handle custom pipeline logic if provided
    custom_pipeline = f"src/pipelines/{pipeline}.py" if pipeline else None

    # Load and return the pipeline
    return pipeline_class.from_pretrained(model, custom_pipeline=custom_pipeline).to(device)


def create_arguments(pipeline_name, classifier, diffusion_settings):
    """Get arguments for the diffusion pipeline. Currently only for guidance pipeline."""
    if pipeline_name == "guidance":
        return {
            "classifier": classifier,
            "alpha": diffusion_settings["args"]["alpha"],
            "guidance_type": diffusion_settings["args"]["guidance"],
        }
    return {}


def visualize_synthetic_images(classifier, sampled_images, sampled_probs):
    """Visualize the synthetic images in a grid. If a classifier is provided, also take that into account."""
    # log the sample grid
    if classifier and classifier.get_n_classes() == 2:
        # specific grid for binary classification -> same as GASTeN
        grid = create_probability_grid(sampled_images, sampled_probs)
    else:
        # generic grid
        grid = create_image_grid(sampled_images)

        # TODO implement a new multi-class classification visualization

    return grid


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
    if configuration["classifier"] is not None:
        classifier = create_classifier(
            configuration["classifier"]["lib"],
            configuration["classifier"]["name"],
            configuration["dataset"]["name"],
            configuration["dataset"]["n_classes"],
            device,
        )

    # get diffusion pipeline
    pipe = create_pipeline(
        diffusion_settings["type"], diffusion_settings["name"], diffusion_settings["pipeline"], device
    )
    # get arguments for the pipeline
    args = create_arguments(diffusion_settings["pipeline"], classifier, diffusion_settings)

    # generate images
    images = pipe(
        generator=torch.Generator().manual_seed(configuration["seed"]),
        num_inference_steps=diffusion_settings["args"]["num-inference-steps"],
        batch_size=diffusion_settings["args"]["batch-size"],
        **args,
    ).images

    # EVALUATION

    # sample images for visualization
    n = min(len(images), configuration["evaluation"]["viz-sample"])
    sampled_images = random.sample(images, n)
    # probabilities if a classifier is provided
    sampled_probs = classifier.predict_from_pil(sampled_images) if classifier else None

    grid = visualize_synthetic_images(classifier, sampled_images, sampled_probs)
    wandb.log({"sample_grid": wandb.Image(grid)})

    if classifier:
        # for the sampled images
        results = label_synthetic_images(classifier, sampled_probs)
        wandb.log({"sample_results": json.dumps(results, indent=4)})

        # for the whole dataset
        # TODO other synthetic images evaluation (using the classifier) - e.g. distributions

    # TODO syntetic images validation (use pydmda)

    # finish wandb
    wandb.finish()


if __name__ == "__main__":
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
