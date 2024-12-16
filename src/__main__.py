"""This is the main file for the diffusion-boundary package."""

import argparse
import json
import os

import torch
import wandb
from diffusers import DDIMPipeline, DDPMPipeline, DiffusionPipeline, PNDMPipeline
from dotenv import load_dotenv

from src.classifier.factory import ClassifierFactory
from src.dataset.aux import get_tst_dataset
from src.dataset.factory import DatasetFactory
from src.evaluation import (
    calculate_fid_metric,
    calculate_synthetic_metrics,
    sample_synthetic_images,
)
from src.utils import generate_group_name, generate_run_id, load_configurations


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


def main(configuration):
    """Generate a sample image."""
    diffusion_settings = configuration["diffusion"]

    # init wandb
    wandb.init(
        project=configuration["project"],
        group=generate_group_name(configuration),
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
        classifier = ClassifierFactory.model_from_lib(
            configuration["classifier"]["lib"],
            configuration["classifier"]["name"],
            configuration["device"],
        )

    # get diffusion pipeline
    pipe = create_pipeline(
        diffusion_settings["type"], diffusion_settings["name"], diffusion_settings["pipeline"], configuration["device"]
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

    # create synthetic dataset
    synth_dataset = DatasetFactory.dataset_from_lib(
        configuration["classifier"]["lib"],
        configuration["classifier"]["name"],
        configuration["dataset"]["name"],
        configuration["dataset"]["n_classes"],
        images,
        configuration["device"],
    )

    # create real dataset with same configs for evaluation purposes
    real_images = get_tst_dataset(
        configuration["dataset"]["name"],
        configuration["dataset"]["subset"],
        diffusion_settings["args"]["batch-size"],
    )
    real_dataset = DatasetFactory.dataset_from_lib(
        configuration["classifier"]["lib"],
        configuration["classifier"]["name"],
        configuration["dataset"]["name"],
        configuration["dataset"]["n_classes"],
        real_images,
        configuration["device"],
    )

    # EVALUATION

    # grid and probs for sampled images
    grid, results = sample_synthetic_images(
        synth_dataset, configuration["evaluation"]["viz-sample-size"], classifier, configuration["dataset"]["subset"]
    )
    wandb.log({"sample_grid": wandb.Image(grid)})
    wandb.log({"sample_results": json.dumps(results, indent=4)})

    # quality metrics (Improived precision, Improved Recall, Density and Coverage)
    metrics, viz = calculate_synthetic_metrics(real_dataset, synth_dataset, configuration["device"])
    wandb.log(metrics)
    wandb.log({"umap": wandb.Image(viz)})

    # FID score (needs a different feature extractor)
    fid_value = calculate_fid_metric(real_dataset, synth_dataset, configuration["device"])
    wandb.log({"FID_score": fid_value})

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
