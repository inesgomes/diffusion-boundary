"""Utility functions for the project."""

import os
import random
import sys
from datetime import datetime, timezone

import yaml

from src.pipelines.schedules import GUIDANCE_SCHEDULES

# schedulers that can be selected in the configuration file (only used by the "sd" pipelines)
SCHEDULERS = ("klms", "ddim")


def generate_run_id():
    """Generate a unique run id based on the current time."""
    current_time = datetime.now(timezone.utc)
    seconds_since_midnight = current_time.hour * 3600 + current_time.minute * 60 + current_time.second
    return f"{current_time.strftime('%Y-%m-%d')}T{seconds_since_midnight}"


def set_configuration_default_values(config):
    """Set default values for missing configuration parameters."""
    # manual vs random seed
    if "seed" not in config["user-args"]:
        config["user-args"]["seed"] = random.randint(1, 100)

    # save images and datasets locally
    if "save-images-disk" not in config["user-args"]:
        config["user-args"]["save-images-disk"] = False
    if "save-metrics-disk" not in config["user-args"]:
        config["user-args"]["save-metrics-disk"] = False

    # check RGB display (only false for grayscale datasets)
    if "display-rgb" not in config["user-args"]:
        config["user-args"]["display-rgb"] = True

    # dataset split (in imagenet is train, otherwise test)
    if "split" not in config["dataset"]:
        config["dataset"]["split"] = "test"

    # check if the mc-dropout should be used
    if "mc-dropout" not in config["evaluation"]:
        config["evaluation"]["mc-dropout"] = {}
        config["evaluation"]["mc-dropout"]["n-samples"] = None
        config["evaluation"]["mc-dropout"]["threshold"] = None

    # default value for certainty threshold (KDN metric and visualizations)
    if "certainty-threshold" not in config["evaluation"]:
        config["evaluation"]["certainty-threshold"] = 0.8

    return config


def resolve_images_path(images_path):
    """Resolve the path of images generated beforehand, relative paths are inside FILESDIR."""
    if not images_path:
        return None

    if not os.path.isabs(images_path):
        images_path = os.path.join(os.getenv("FILESDIR", ""), images_path)

    # fail before loading models and starting the wandb run
    if not os.path.isfile(images_path):
        print(f"Images file {images_path} not found.")
        sys.exit(1)

    return images_path


def prepare_diffusion_grid(diffusion_config):
    """Transform the grid parameters to lists and validate the scheduler and guidance schedule names."""
    # transform certain arguments to list
    for key in ("guidance", "alpha", "guidance-scale", "guidance-freq", "guidance-schedule"):
        if not isinstance(diffusion_config["args"][key], list):
            diffusion_config["args"][key] = [diffusion_config["args"][key]]
    if not isinstance(diffusion_config["scheduler"], list):
        diffusion_config["scheduler"] = [diffusion_config["scheduler"]]

    # the scheduler drives the x_0 prediction used by the guidance, so only known ones are accepted
    unknown = [name for name in diffusion_config["scheduler"] if name not in SCHEDULERS]
    if unknown:
        print(f"Unknown scheduler(s) {unknown}. Valid options are {list(SCHEDULERS)}.")
        sys.exit(1)

    # a typo here would otherwise only fail after the models are loaded and the run has started
    unknown = [name for name in diffusion_config["args"]["guidance-schedule"] if name not in GUIDANCE_SCHEDULES]
    if unknown:
        print(f"Unknown guidance schedule(s) {unknown}. Valid options are {list(GUIDANCE_SCHEDULES)}.")
        sys.exit(1)

    return diffusion_config


def load_configurations(config_path):
    """Load configuration file from path and modify accondingly."""
    try:
        with open(config_path, encoding="utf-8") as file:
            # load configuration file
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Config file {config_path} not found.")
        sys.exit(1)

    # unit tests
    if "name" not in config["dataset"]:
        print("Dataset name must be defined in the configuration file.")
        sys.exit(1)

    # check if dataset subset exist
    if "subset" not in config["dataset"]:
        config["dataset"]["subset"] = None
        if "classes" in config["diffusion"]["args"]:
            config["dataset"]["subset"] = config["diffusion"]["args"]["classes"]

    # check if classifier exist and configure path
    if "classifier" not in config:
        config["classifier"] = None
    else:
        if config["classifier"]["lib"] == "local":
            config["classifier"]["name"] = os.getenv("MODELS_DIR") + "/" + config["classifier"]["name"]
        if "corrupt" not in config["classifier"]:
            config["classifier"]["corrupt"] = 0
        if "calibrate" not in config["classifier"]:
            config["classifier"]["calibrate"] = False

    # check if guidance exists
    if "pipeline" not in config["diffusion"]:
        config["diffusion"]["pipeline"] = None
    if "type" not in config["diffusion"]:
        config["diffusion"]["type"] = None
    if "scheduler" not in config["diffusion"]:
        config["diffusion"]["scheduler"] = "klms"
    if "guidance" not in config["diffusion"]["args"]:
        config["diffusion"]["args"]["guidance"] = "noguidance"
    if "negative-prompt" not in config["diffusion"]["args"]:
        config["diffusion"]["args"]["negative-prompt"] = ""
    # gamma says how many guidance updates there are, this says where they land
    if "guidance-schedule" not in config["diffusion"]["args"]:
        config["diffusion"]["args"]["guidance-schedule"] = "throughout"

    # images generated beforehand (e.g. the BigGAN baseline) are loaded instead of being generated
    config["diffusion"]["images-path"] = resolve_images_path(config["diffusion"].get("images-path"))

    config["diffusion"] = prepare_diffusion_grid(config["diffusion"])

    # set default values for missing configuration parameters
    config = set_configuration_default_values(config)

    return config
