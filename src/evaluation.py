"""Evaluation module. Contains methods to evaluate the synthetic images."""

import math

import matplotlib.pyplot as plt
import torch
from PIL import Image
from pymdma.image.measures.synthesis_val import (
    Coverage,
    Density,
    FrechetDistance,
    ImprovedPrecision,
    ImprovedRecall,
)
from pymdma.image.models.features import ExtractorFactory
from umap import UMAP

from src.dataset.aux import LABELS


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


def sample_synthetic_images(synth_dataset, sample_size, classifier, subset_labels=None):
    """Visualize the synthetic images in a grid. If a classifier is provided, also take that into account."""
    # sample images for visualization
    sampled_tensors, sampled_images = synth_dataset.sample_as_tensor(sample_size)

    # and probabilities if a classifier is provided
    sampled_tensors = sampled_tensors.to(synth_dataset.get_device())
    sampled_probs = classifier.predict(sampled_tensors) if classifier else None

    # log the sample grid
    if classifier and synth_dataset.get_n_classes() == 2:
        # specific grid for binary classification -> same as GASTeN
        grid = create_probability_grid(sampled_images, sampled_probs)
    else:
        # generic grid
        grid = create_image_grid(sampled_images)
        # TODO implement a new multi-class classification visualization

    results = None
    if classifier:
        # for the sampled images, provide the top probabilities and labels (textual information)
        labels = subset_labels if subset_labels is not None else LABELS[synth_dataset.get_dataset_name()]
        results = label_synthetic_images(labels, synth_dataset.get_n_classes(), sampled_probs)

        # for the whole dataset
        # TODO other synthetic images evaluation (using the classifier) - e.g. distributions

    return grid, results


def label_synthetic_images(labels, n_classes, probabilities):
    """Label the images using the classifier and return the results."""
    # if binary classification, prepare the results
    if n_classes == 2:
        probabilities = torch.stack([probabilities, 1 - probabilities], dim=1)
    # get the top probabilities, indices and respective labels for each image (logging purposes)
    top_probs, top_indices = torch.topk(probabilities, k=n_classes, dim=1)

    # TODO: this is not great, probably should be done in the classifier or dataset classes
    return {
        i: {labels[int(idx)]: round(prob.item(), 2) for idx, prob in zip(top_indices[i], top_probs[i])}
        for i in range(top_indices.size(0))
    }


def calculate_synthetic_metrics(real_dataset, synth_dataset):
    """Calculate synthetic validation metrics from pymdma library.

    TODO: missing tunning the k values for the metrics.
    """
    # extract features to compute quality metrics
    extractor = ExtractorFactory.model_from_name(name="dino_vits8")
    real_features = extractor(real_dataset.image_to_tensor()).detach().cpu().numpy()
    synth_features = extractor(synth_dataset.image_to_tensor()).detach().cpu().numpy()

    ip_result = ImprovedPrecision(k=6).compute(real_features=real_features, fake_features=synth_features)
    # precision_dataset, _ = ip_result.value

    ir_result = ImprovedRecall(k=6).compute(real_features=real_features, fake_features=synth_features)
    # recall_dataset, _ = ir_result.value

    density_result = Density(k=6).compute(real_features=real_features, fake_features=synth_features)
    # density_dataset, _ = density_result.value

    coverage_result = Coverage(k=6).compute(real_features=real_features, fake_features=synth_features)
    # coverage_dataset, _ = coverage_result.value

    results_dict = {
        "precision": ip_result.value[0],
        "recall": ir_result.value[0],
        "density": density_result.value[0],
        "coverage": coverage_result.value[0],
    }

    # UMAP 2D visualization
    fig = umap_visualization(real_features, synth_features)

    return results_dict, fig


def calculate_fid_metric(real_dataset, synth_dataset):
    """Calculate the Frechet Inception Distance (FID) between two datasets using the implementation from pymdma library."""
    extractor = ExtractorFactory.model_from_name(name="inception_fid")

    real_features = extractor(real_dataset.image_to_norm_tensor()).detach().cpu().numpy()
    synth_features = extractor(synth_dataset.image_to_norm_tensor()).detach().cpu().numpy()

    fid_result = FrechetDistance().compute(real_features=real_features, fake_features=synth_features)

    return fid_result.value[0]


def umap_visualization(real_features, synth_features):
    """2D UMAP visualization of the features. Returns the figure."""
    umap = UMAP(n_components=2, random_state=10, n_jobs=1)
    real_feats_2d = umap.fit_transform(real_features)
    fake_feats_2d = umap.transform(synth_features)

    fig = plt.figure(figsize=(10, 10))
    plt.scatter(real_feats_2d[:, 0], real_feats_2d[:, 1], s=3, label="Real Samples", color="red")
    plt.scatter(fake_feats_2d[:, 0], fake_feats_2d[:, 1], s=3, label="Fake Samples", color="blue")
    plt.title("UMAP Features Visualization | Real vs Synthetic")
    plt.legend()
    return fig
