"""This is the main file for the diffusion-boundary package."""

import argparse
import os

import torch
import wandb
from dotenv import load_dotenv

from src.classifier import get_timm_classifier
from src.diffusion import get_custom_pipe, get_pipe
from src.utils import create_image_grid, generate_run_id, load_configurations


def main(configuration):
    """Generate a sample image."""
    diffusion_settings = configuration["diffusion"]
    device = configuration["device"]

    # init wandb
    wandb.init(
        project=configuration["project"],
        group=configuration["name"],
        job_type=diffusion_settings["type"],
        entity=os.getenv("ENTITY"),
        name=generate_run_id(),
        config={
            "seed": configuration["seed"],
            "diffusion": diffusion_settings,
            "classsifier": configuration["classifier"],
        },
    )

    # TODO: i dont like this, maybe I need to refactor the code
    args = {}
    if "pipeline" in diffusion_settings:
        # get custom pipeline
        pipe = get_custom_pipe(
            diffusion_settings["type"], diffusion_settings["name"], diffusion_settings["pipeline"], device
        )
        if configuration["classifier"] is not None:
            # TODO: only working for timm lib
            # get classifier
            classifier, preprocessing = get_timm_classifier(configuration["classifier"]["name"], device)
            args = {"classifier": classifier, "preprocessing": preprocessing, "alpha": diffusion_settings["alpha"]}
    else:
        # get default pipeline
        pipe = get_pipe(diffusion_settings["type"], diffusion_settings["name"], device)

    # create generator
    generator = torch.Generator(device=device).manual_seed(configuration["seed"])

    out = pipe(
        generator=generator,
        num_inference_steps=diffusion_settings["num-inference-steps"],
        batch_size=diffusion_settings["batch-size"],
        **args,
    )
    # image = out[0]
    # grid = make_grid_v1(out)

    # TODO make predictions for the images and check the probabilities and entropy

    # Create a grid from the batch
    grid = create_image_grid(out, max_columns=10)

    # Log the image
    wandb.log({"sample_grid": wandb.Image(grid)})

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
