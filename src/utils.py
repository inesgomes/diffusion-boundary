"""Utility functions for the project."""

import os
import random
import sys
from datetime import datetime

import yaml


def generate_run_id():
    """Generate a unique run id based on the current time."""
    current_time = datetime.utcnow()
    seconds_since_midnight = current_time.hour * 3600 + current_time.minute * 60 + current_time.second
    return f"{current_time.strftime('%Y-%m-%d')}T{seconds_since_midnight}"


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

    # save images and datasets locally
    if "save-disk" not in config:
        config["save-disk"] = False

    # manual vs random seed
    if "seed" not in config:
        config["seed"] = random.randint(1, 100)

    # check if classifier exist
    if "classifier" not in config:
        config["classifier"] = None
    else:
        if config["classifier"]["lib"] == "local":
            config["classifier"]["name"] = os.getenv("MODELS_DIR") + "/" + config["classifier"]["name"]
        if "corrupt" not in config["classifier"]:
            config["classifier"]["corrupt"] = 0

    # check if guidance exists
    if "pipeline" not in config["diffusion"]:
        config["diffusion"]["pipeline"] = None
    if "type" not in config["diffusion"]:
        config["diffusion"]["type"] = None
    if "guidance" not in config["diffusion"]["args"]:
        config["diffusion"]["args"]["guidance"] = "noguidance"
    if "negative-prompt" not in config["diffusion"]["args"]:
        config["diffusion"]["args"]["negative-prompt"] = ""

    # check if the mc-dropout should be used
    if "mc-dropout" not in config["evaluation"]:
        config["evaluation"]["mc-dropout"] = {}
        config["evaluation"]["mc-dropout"]["n-samples"] = None
        config["evaluation"]["mc-dropout"]["threshold"] = None

    # default value
    if "certainty-threshold" not in config["evaluation"]:
        config["evaluation"]["certainty-threshold"] = 0.8

    # transform certain arguments to list
    if not isinstance(config["diffusion"]["args"]["guidance"], list):
        config["diffusion"]["args"]["guidance"] = [config["diffusion"]["args"]["guidance"]]
    if not isinstance(config["diffusion"]["args"]["alpha"], list):
        config["diffusion"]["args"]["alpha"] = [config["diffusion"]["args"]["alpha"]]
    if not isinstance(config["diffusion"]["args"]["guidance-scale"], list):
        config["diffusion"]["args"]["guidance-scale"] = [config["diffusion"]["args"]["guidance-scale"]]

    # check if dataset subset exist
    if "subset" not in config["dataset"]:
        config["dataset"]["subset"] = None
        if "classes" in config["diffusion"]["args"]:
            config["dataset"]["subset"] = config["diffusion"]["args"]["classes"]

    if "split" not in config["dataset"]:
        config["dataset"]["split"] = "test"

    # check RGB display
    if "display-rgb" not in config["evaluation"]:
        config["evaluation"]["display-rgb"] = True

    return config
