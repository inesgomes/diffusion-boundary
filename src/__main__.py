"""This is the main file for the diffusion-boundary package."""

import argparse
import math
import os

import pandas as pd
import torch
import wandb
from diffusers import DDIMPipeline, DDPMPipeline, DiffusionPipeline, PNDMPipeline
from dotenv import load_dotenv
from tqdm import tqdm

from src.classifier.factory import ClassifierFactory
from src.dataset.aux import get_tst_dataset_streaming
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
    return pipeline_class.from_pretrained(
        model, custom_pipeline=custom_pipeline, cache_dir=os.getenv("HF_MODELS_CACHE")
    ).to(device)


def create_arguments(pipeline_name, pipeline_type, classifier, dataset, diffusion_arguments):
    """Get arguments for the diffusion pipeline. Currently only for guidance pipeline."""
    args = {}
    if pipeline_name == "guidance":
        args.update(
            {
                "classifier": classifier,
                "transformation": dataset,
                "alpha": diffusion_arguments["alpha"],
                "guidance_type": diffusion_arguments["guidance"],
                "guidance_freq": diffusion_arguments["guidance-freq"],
            }
        )
    if pipeline_type == "text-to-image":
        # TODO add latents and guidance_scale
        args.update({"prompt": diffusion_arguments["prompt"]})
    return args


def generate_images(diffusion_settings, classifier, dataset, num_images, batch_size, seed, device):
    """Generate images using the diffusion pipeline described in the config file."""
    # get diffusion pipeline
    pipe = create_pipeline(
        diffusion_settings["type"], diffusion_settings["name"], diffusion_settings["pipeline"], device
    )
    # get arguments for the pipeline
    args = create_arguments(
        diffusion_settings["pipeline"], diffusion_settings["type"], classifier, dataset, diffusion_settings["args"]
    )

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
        job_type=diffusion_settings["args"]["guidance"],
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

    # prepare the original dataset, for evaluation purposes, with the same number of samples as the generated ones
    real_images, real_labels, class_labels = get_tst_dataset_streaming(
        configuration["dataset"]["name"],
        configuration["dataset"]["split"],
        configuration["evaluation"]["num-images"],
        configuration["dataset"]["subset"],
    )
    real_dataset = DatasetFactory.dataset_from_lib(
        configuration["classifier"]["lib"],
        configuration["classifier"]["name"],
        configuration["dataset"]["name"],
        configuration["dataset"]["n_classes"],
        class_labels,
        real_images,
    )
    real_dataset_res = prepare_dataset_results(
        real_dataset,
        classifier,
        configuration["batch-size"],
        configuration["device"],
        real_labels,
    )

    torch.cuda.empty_cache()

    # prepare synthetic dataset object
    synth_dataset = DatasetFactory.dataset_from_lib(
        configuration["classifier"]["lib"],
        configuration["classifier"]["name"],
        configuration["dataset"]["name"],
        configuration["dataset"]["n_classes"],
        class_labels,
        None,
    )

    # generate images
    images = generate_images(
        diffusion_settings,
        classifier,
        synth_dataset,
        configuration["evaluation"]["num-images"],
        configuration["batch-size"],
        configuration["seed"],
        configuration["device"],
    )
    synth_dataset.set_images(images)

    torch.cuda.empty_cache()

    # prepare results, from synthetic dataset
    synth_dataset_res = prepare_dataset_results(
        synth_dataset,
        classifier,
        configuration["batch-size"],
        configuration["device"],
    )

    # _row = synth_dataset_res.iloc[0]
    # print(_row)
    # image_tensor = (synth_dataset[_row["image_id"]] + 1) / 2
    # image_tensor = torch.clamp(image_tensor, 0, 1)
    # image_np = image_tensor.permute(1, 2, 0).numpy()

    # fig, ax = plt.subplots(figsize=(1.5, 2.5))
    # ax.imshow(image_np, cmap="gray")

    # wandb.log({"fig_test": wandb.Image(fig)})

    # save if needed
    if configuration["log"]["images"]:
        path = os.getenv("FILESDIR") + "/logs/" + wandb.run.id
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/images.pkl", "wb") as f:
            torch.save(images, f)
            print("Images saved at", path)

    # EVALUATION of the synthetic dataset

    # from features

    # quality metrics (Improved precision, Improved Recall, Density and Coverage)
    metrics, features_umap = calculate_synthetic_metrics(
        real_dataset,
        synth_dataset,
        configuration["batch-size"],
        configuration["device"],
    )
    wandb.log(metrics)
    wandb.log({"umap": wandb.Image(features_umap)})

    # FID score (calculated seperatly because it needs a different feature extractor)
    fid_value = calculate_fid_metric(real_dataset, synth_dataset, configuration["batch-size"], configuration["device"])
    wandb.log({"FID_score": fid_value})

    # from probabilities

    # distributions (boxplot): metric and classes
    dist_metric, dist_probs = visualize_distributions(
        real_dataset_res, synth_dataset_res, configuration["dataset"]["n_classes"]
    )
    wandb.log({"dist_metrics": wandb.Image(dist_metric)})

    # visualize confusion matrix, only if the number of classes allows it
    if synth_dataset.get_n_classes() <= 20:
        # probabilities per label
        wandb.log({"dist_labels": wandb.Image(dist_probs)})

        viz_pairs, table_confusion = visualize_confusion(
            real_dataset_res,
            synth_dataset_res,
            configuration["dataset"]["n_classes"],
            configuration["evaluation"]["certainty-threshold"],
        )
        wandb.log({"pairs_cm": wandb.Image(viz_pairs)})
        wandb.log({"_boundaries": wandb.Table(dataframe=table_confusion)})

    # sample: grid of images and respective probs
    # entropy is default metric
    sort_metric = diffusion_settings["args"]["guidance"] if diffusion_settings["pipeline"] == "guidance" else "entropy"
    grid, results = visualize_sample_synthetic_images(
        synth_dataset,
        synth_dataset_res,
        configuration["evaluation"]["viz-sample-size"],
        sort_metric,
        configuration["evaluation"]["display-rgb"],
    )

    wandb.log({"sample_grid": wandb.Image(grid)})
    wandb.log({"_sample_probabilities": wandb.Table(dataframe=results)})

    # finish wandb
    wandb.finish()


if __name__ == "__main__":
    # load environment variables
    load_dotenv()

    # enable torch32 for faster inference
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Set display options for Pandas
    pd.set_option("display.max_rows", None)  # Show all rows
    pd.set_option("display.max_columns", None)  # Show all columns
    pd.set_option("display.width", None)  # Expand the width to fit the data

    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", required=True, help="Configuration file")
    my_args = parser.parse_args()
    # load information from config file
    config = load_configurations(my_args.config_path)

    # execute code
    main(config)
