"""This is the main file for the diffusion-boundary package."""

import argparse
import os
import sys

import torch
import wandb
import yaml
from dotenv import load_dotenv

from src.classifier import get_classifier
from src.diffusion import get_custom_pipe


def generate_sample(pipe, classifier, preprocessing, diffusion_settings, device):
    """Generate a sample image using the pipeline specified."""
    generator = torch.Generator(device=device).manual_seed(42)
    out = pipe(
        generator=generator,
        classifier=classifier,
        preprocessing=preprocessing,
        num_inference_steps=diffusion_settings["num_inference_steps"],
        alpha=diffusion_settings["alpha"],
        eta=diffusion_settings["eta"],
    )

    return out[0]


def main(configuration):
    """Generate a sample image."""
    diffusion_settings = configuration["diffusion"]
    device = configuration["device"]

    # init wandb
    wandb.init(
        project=configuration["project"],
        group=configuration["name"],
        job_type=configuration["job"],
        entity=os.getenv("ENTITY"),
        # name='', # maybe later can be useful
        config={
            "seed": 42,
            "alpha": diffusion_settings["alpha"],
            "eta": diffusion_settings["eta"],
            "num_inference_steps": diffusion_settings["num-inference-steps"],
        },
    )

    # get classifier and pipe
    pipe = get_custom_pipe(device)
    classifier, preprocessing = get_classifier(device)

    # generate sample and save
    image = generate_sample(pipe, classifier, preprocessing, diffusion_settings, device)

    # Log the image
    wandb.log({"sample_image": wandb.Image(image)})
    # image.save("samples/tst.png")

    wandb.finish()


if __name__ == "__main__":
    # TODO: method should be able to receive my trained classifier

    # load environment variables
    load_dotenv()

    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", required=True, help="Configuration file")
    args = parser.parse_args()

    try:
        with open(args.config_path, encoding="utf-8") as file:
            config = yaml.safe_load(file)
            main(config)
    except FileNotFoundError:
        print(f"Config file {args.config_path} not found.")
        sys.exit(1)
