"""This is the main file for the diffusion-boundary package."""

import argparse
import math
import os

import torch
import wandb
from diffusers import DDIMPipeline, DDPMPipeline, DiffusionPipeline, PNDMPipeline
from dotenv import load_dotenv
from tqdm import tqdm

from src.classifier.factory import ClassifierFactory
from src.dataset.aux import get_tst_dataset
from src.dataset.factory import DatasetFactory
from src.evaluation import (
    calculate_fid_metric,
    calculate_synthetic_metrics,
    prepare_dataset_results,
    visualize_confusion,
    visualize_distributions,
    visualize_sample_synthetic_images,
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
            "guidance_freq": diffusion_settings["args"]["guidance-freq"],
        }
    return {}


def generate_images(diffusion_settings, classifier, num_images, batch_size, seed, device):
    """Generate images using the diffusion pipeline described in the config file."""
    # get diffusion pipeline
    pipe = create_pipeline(
        diffusion_settings["type"], diffusion_settings["name"], diffusion_settings["pipeline"], device
    )
    # get arguments for the pipeline
    args = create_arguments(diffusion_settings["pipeline"], classifier, diffusion_settings)

    # generate images in batches
    num_batches = math.ceil(num_images / batch_size)
    images = []
    for _ in tqdm(range(num_batches), desc="Generating images"):
        batch_size_to_use = min(batch_size, num_images - len(images))
        batch_images = pipe(
            generator=torch.Generator().manual_seed(seed),
            num_inference_steps=diffusion_settings["args"]["num-inference-steps"],
            batch_size=batch_size_to_use,
            **args,
        ).images
        images.extend(batch_images)

    print(f"Generated {len(images)} images")
    return images


def main(configuration):
    """Generate a sample image."""
    diffusion_settings = configuration["diffusion"]

    # init wandb
    wandb.init(
        project=configuration["project"],
        group=generate_group_name(configuration),
        job_type=diffusion_settings["type"],  # TODO: guidance
        entity=os.getenv("ENTITY"),
        name=generate_run_id(),
        config={
            "seed": configuration["seed"],
            "diffusion": diffusion_settings,
            "classsifier": configuration["classifier"]["name"],
            "log_images": configuration["log"]["images"],
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

    # generate images
    images = generate_images(
        diffusion_settings,
        classifier,
        configuration["evaluation"]["num-images"],
        configuration["batch-size"],
        configuration["seed"],
        configuration["device"],
    )

    # save if needed
    if configuration["log"]["images"]:
        path = os.getenv("FILESDIR") + "/logs/" + wandb.run.id
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/images.pkl", "wb") as f:
            torch.save(images, f)
            print("Images saved at", path)

    # create synthetic dataset
    synth_dataset = DatasetFactory.dataset_from_lib(
        configuration["classifier"]["lib"],
        configuration["classifier"]["name"],
        configuration["dataset"]["name"],
        configuration["dataset"]["n_classes"],
        images,
    )
    # create real dataset with same configs for evaluation purposes
    real_images, real_labels = get_tst_dataset(
        configuration["dataset"]["name"],
        configuration["dataset"]["subset"],
        configuration["evaluation"]["num-images"],
    )
    real_dataset = DatasetFactory.dataset_from_lib(
        configuration["classifier"]["lib"],
        configuration["classifier"]["name"],
        configuration["dataset"]["name"],
        configuration["dataset"]["n_classes"],
        real_images,
    )

    # EVALUATION of the synthetic dataset

    # from features

    # FID score (calculated seperatly because it needs a different feature extractor)
    fid_value = calculate_fid_metric(real_dataset, synth_dataset, configuration["batch-size"], configuration["device"])
    wandb.log({"FID_score": fid_value})

    # quality metrics (Improved precision, Improved Recall, Density and Coverage)
    metrics, features_umap = calculate_synthetic_metrics(
        real_dataset,
        synth_dataset,
        configuration["batch-size"],
        configuration["device"],
    )
    wandb.log(metrics)
    wandb.log({"umap": wandb.Image(features_umap)})

    # from probabilities

    real_dataset_res = prepare_dataset_results(
        real_dataset,
        classifier,
        diffusion_settings["args"]["guidance"],
        configuration["batch-size"],
        configuration["device"],
    )
    synth_dataset_res = prepare_dataset_results(
        synth_dataset,
        classifier,
        diffusion_settings["args"]["guidance"],
        configuration["batch-size"],
        configuration["device"],
    )

    # distributions (boxplot): metric and classes
    dist_metric, dist_probs = visualize_distributions(
        real_dataset_res, synth_dataset_res, diffusion_settings["args"]["guidance"]
    )
    wandb.log({f"dist_{diffusion_settings['args']['guidance']}": wandb.Image(dist_metric)})
    wandb.log({"dist_labels": wandb.Image(dist_probs)})

    # visualize confusion matrix
    viz_pairs, viz_diff = visualize_confusion(
        real_dataset_res,
        synth_dataset_res,
        diffusion_settings["args"]["guidance"],
        configuration["evaluation"]["certainty-threshold"],
    )
    wandb.log({"pairs_cm": wandb.Image(viz_pairs)})
    wandb.log({"boundary_cm": wandb.Image(viz_diff)})

    # sample: grid of images and respective probs
    grid, results = visualize_sample_synthetic_images(
        synth_dataset,
        configuration["evaluation"]["viz-sample-size"],
        classifier,
        diffusion_settings["args"]["guidance"],
        configuration["evaluation"]["certainty-threshold"],
        configuration["device"],
    )
    wandb.log({"sample_grid": wandb.Image(grid)})
    wandb.log({"_sample_probabilities": wandb.Table(dataframe=results)})

    # classifier evaluation (for debugging purposes)
    if classifier:
        real_dataset_res["label"] = real_labels
        real_dataset_res["prediction"] = real_dataset_res.iloc[:, 1:-1].values.argmax(axis=1)

        accuracy = (real_dataset_res["label"] == real_dataset_res["prediction"]).sum() / len(real_dataset_res)
        print(f"Accuracy: {accuracy:.2%}")

    # finish wandb
    wandb.finish()


if __name__ == "__main__":
    # load environment variables
    load_dotenv()

    # enable torch32 for faster inference
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", required=True, help="Configuration file")
    my_args = parser.parse_args()
    # load information from config file
    config = load_configurations(my_args.config_path)

    # execute code
    main(config)
