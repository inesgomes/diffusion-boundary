"""Utility functions for the project."""

import random
import sys
from datetime import datetime

import numpy as np
import yaml
from PIL import Image


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
    if "pipeline" not in config["diffusion"]:
        config["diffusion"]["pipeline"] = "default"

    # manual vs random seed
    if "seed" not in config:
        config["seed"] = random.randint(1, 100)

    # unit tests
    if "name" not in config["dataset"]:
        print("Dataset name must be defined in the configuration file.")
        sys.exit(1)

    # TODO confirm that some values are not None or do not exist
    return config


def create_image_grid(image_numpy, max_columns=10):
    """
    Create a grid image from a batch of images.

    Args:
        image_numpy: NumPy array of shape (B, H, W, C).
        max_columns: Maximum number of columns in the grid.
    Returns:
        A single grid image in PIL format.
    """
    batch_size, height, width, channels = image_numpy.shape
    nrows = (batch_size + max_columns - 1) // max_columns  # Calculate required rows
    ncols = min(batch_size, max_columns)  # Limit columns to max_columns

    # Create a blank canvas for the grid
    grid = np.zeros((nrows * height, ncols * width, channels), dtype=image_numpy.dtype)

    # Populate the grid
    for idx, img in enumerate(image_numpy):
        row = idx // ncols
        col = idx % ncols
        grid[row * height : (row + 1) * height, col * width : (col + 1) * width, :] = img

    # Convert back to 0-255 range
    grid = (grid * 255).astype(np.uint8)

    if channels == 1:  # 1-channel (Grayscale)
        return Image.fromarray(grid.squeeze(-1), mode="L")  # "L" for grayscale
    # 3-channel (RGB)
    return Image.fromarray(grid, mode="RGB")
