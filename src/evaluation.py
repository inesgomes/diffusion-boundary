"""Evaluation module. Contains methods to evaluate the synthetic images."""

import math

import torch
from PIL import Image

from src.dataset import get_labels


def create_probability_grid(images, probabilities, n_columns=10):
    """Create a grid of images with probabilities.

    The images are grouped into bins acording to their probabilities. We are assuming that the probabilities are in the same order as the images.
    """
    # Get the size and mode from the first image
    img_size = images[0].size
    img_mode = images[0].mode

    # Group images into bins based on their probabilities
    bins = [[] for _ in range(n_columns)]
    for img, prob in zip(images, probabilities):
        bin_index = min(math.floor(prob * n_columns), n_columns - 1)
        bins[bin_index].append(img)

    # Prepare the grid image
    max_rows = max(len(bin) for bin in bins)
    grid_image = Image.new(img_mode, (n_columns * img_size[0], max_rows * img_size[1]), color="black")

    # Paste images into the grid
    for col, bin_images in enumerate(bins):
        for row, img in enumerate(bin_images):
            grid_image.paste(img, (col * img_size[0], row * img_size[1]))

    return grid_image


def create_image_grid(images, n_columns=10):
    """Create a grid of images with a fixed number of columns."""
    # Get the size and mode from the first image (assuming all images have the same size and mode)
    img_size = images[0].size
    img_mode = images[0].mode

    # Calculate the number of rows needed for the grid
    n_images = len(images)
    n_rows = math.ceil(n_images / n_columns)

    # Create a blank canvas for the grid (black background)
    grid_image = Image.new(img_mode, (n_columns * img_size[0], n_rows * img_size[1]), color="black")

    # Paste each image into the grid at the appropriate location
    for idx, img in enumerate(images):
        row = idx // n_columns  # Calculate row index
        col = idx % n_columns  # Calculate column index
        x_offset = col * img_size[0]
        y_offset = row * img_size[1]
        grid_image.paste(img, (x_offset, y_offset))

    return grid_image


def label_synthetic_images(classifier, probabilities):
    """Label the images using the classifier and return the results."""
    # if binary classification, prepare the results
    n_classes = classifier.get_n_classes()
    if n_classes == 2:
        probabilities = torch.stack([probabilities, 1 - probabilities], dim=1)
    # get the top probabilities, indices and respective labels for each image (logging purposes)
    top_probs, top_indices = torch.topk(probabilities, k=n_classes, dim=1)
    labels = get_labels(classifier.get_dataset_name())
    results_dict = {
        i: {labels[int(idx)]: round(prob.item(), 2) for idx, prob in zip(top_indices[i], top_probs[i])}
        for i in range(top_indices.size(0))
    }
    return results_dict
