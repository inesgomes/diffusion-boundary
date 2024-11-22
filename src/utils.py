"""Utility functions for the project."""

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

    # check if classifier exist
    if "classifier" not in config:
        config["classifier"] = None

    # check if type exist
    if "type" not in config["diffusion"]:
        config["diffusion"]["type"] = "default"

    # manual vs random seed
    if "seed" not in config:
        config["seed"] = random.randint(1, 100)

    # TODO confirm that some values are not None or do not exist
    return config
