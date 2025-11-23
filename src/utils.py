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
    if "guidance" not in config["diffusion"]["args"]:
        config["diffusion"]["args"]["guidance"] = "noguidance"
    if "negative-prompt" not in config["diffusion"]["args"]:
        config["diffusion"]["args"]["negative-prompt"] = ""

    # transform certain arguments to list
    if not isinstance(config["diffusion"]["args"]["guidance"], list):
        config["diffusion"]["args"]["guidance"] = [config["diffusion"]["args"]["guidance"]]
    if not isinstance(config["diffusion"]["args"]["alpha"], list):
        config["diffusion"]["args"]["alpha"] = [config["diffusion"]["args"]["alpha"]]
    if not isinstance(config["diffusion"]["args"]["guidance-scale"], list):
        config["diffusion"]["args"]["guidance-scale"] = [config["diffusion"]["args"]["guidance-scale"]]

    # set default values for missing configuration parameters
    config = set_configuration_default_values(config)

    return config
