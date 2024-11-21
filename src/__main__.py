"""This is the main file for the diffusion-boundary package."""

import argparse
import os
import sys
from datetime import datetime

import torch
import wandb
import yaml
from dotenv import load_dotenv

from src.classifier import get_vit_classifier
from src.diffusion import get_custom_pipe, get_pipe


def generate_run_id():
    """Generate a unique run id based on the current time."""
    current_time = datetime.utcnow()
    seconds_since_midnight = current_time.hour * 3600 + current_time.minute * 60 + current_time.second
    return f"{current_time.strftime('%Y-%m-%d')}T{seconds_since_midnight}"


def generate_sample(pipe, num_inference_steps, batch_size, device, **kwargs):
    """Generate a sample image using the pipeline specified."""
    generator = torch.Generator(device=device).manual_seed(42)

    out = pipe(generator=generator, num_inference_steps=num_inference_steps, batch_size=batch_size, **kwargs)
    print(out)
    return out[0]


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
            "seed": 42,
            "diffusion": diffusion_settings,
            "classsifier": configuration["classifier"],
        },
    )

    # TODO: i dont like this, maybe I need to refactor the code

    args = {}
    if "pipeline" in diffusion_settings:
        # get custom pipeline
        pipe = get_custom_pipe(
            diffusion_settings["type"], diffusion_settings["model"], diffusion_settings["pipeline"], device
        )
        if diffusion_settings["pipeline"] == "guidance":
            # TODO: check which classifier
            # get classifier
            classifier, preprocessing = get_vit_classifier(configuration["classifier"]["model"], device)
            args = {"classifier": classifier, "preprocessing": preprocessing, "alpha": diffusion_settings["alpha"]}
    else:
        # get default pipeline
        pipe = get_pipe(diffusion_settings["type"], diffusion_settings["model"], device)

    # create generator
    generator = torch.Generator(device=device).manual_seed(42)

    out = pipe(
        generator=generator,
        num_inference_steps=diffusion_settings["num-inference-steps"],
        batch_size=diffusion_settings["batch-size"],
        **args,
    )
    image = out[0]

    # Log the image
    wandb.log({"sample_image": wandb.Image(image)})

    wandb.finish()


if __name__ == "__main__":
    # TODO: method should be able to receive my trained classifier

    # load environment variables
    load_dotenv()

    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", required=True, help="Configuration file")
    my_args = parser.parse_args()

    try:
        with open(my_args.config_path, encoding="utf-8") as file:
            # load configuration file
            config = yaml.safe_load(file)
            # check if arguments exist
            if "arguments" not in config["diffusion"]:
                config["diffusion"]["arguments"] = None
            main(config)
    except FileNotFoundError:
        print(f"Config file {my_args.config_path} not found.")
        sys.exit(1)
