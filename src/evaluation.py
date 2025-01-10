"""Evaluation module. Contains methods to evaluate the synthetic images."""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers.utils import make_image_grid
from PIL import Image
from pymdma.image.measures.synthesis_val import (
    Coverage,
    Density,
    FrechetDistance,
    ImprovedPrecision,
    ImprovedRecall,
)
from pymdma.image.models.features import ExtractorFactory
from torch.utils.data import DataLoader
from tqdm import tqdm
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


def sample_synthetic_images(synth_dataset, sample_size, classifier, device):
    """Visualize the synthetic images in a grid. If a classifier is provided, also take that into account."""
    # sample images for visualization
    # temporary fix for MNIST dataset
    synth_dataset.set_convert_rgb(synth_dataset.get_dataset_name() != "mnist")
    sampled_tensors, sampled_images = synth_dataset.sample_to_tensor(sample_size)
    # and probabilities if a classifier is provided
    sampled_probs = classifier.predict(sampled_tensors.to(device)) if classifier else None

    # log the sample grid
    if classifier and synth_dataset.get_n_classes() == 2:
        # specific grid for binary classification -> same as GASTeN
        grid = create_probability_grid(sampled_images, sampled_probs)
    else:
        # generic grid
        n_cols = 10
        n_rows = math.ceil(len(sampled_images) / n_cols)
        grid = make_image_grid(sampled_images, rows=n_rows, cols=n_cols)
        # TODO implement a new multi-class classification visualization

    results = None
    if classifier:
        # for the sampled images, provide the probabilities in a dataframe format
        labels = LABELS[synth_dataset.get_dataset_name()]
        results = label_synthetic_images(labels, synth_dataset.get_n_classes(), sampled_probs)
        # TODO to save as csv

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


def calculate_synthetic_metrics(real_dataset, synth_dataset, device, batch_size=64):
    """Calculate synthetic validation metrics from pymdma library.

    TODO: missing tunning the k values for the metrics.
    """
    # extract features to compute quality metrics
    extractor = ExtractorFactory.model_from_name(name="dino_vits8").to(device)

    # data loader
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    synth_loader = DataLoader(synth_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    # Extract features for real dataset
    real_features_list = []
    for batch in tqdm(real_loader, desc="Processing Real Images -> dino vits8"):
        with torch.no_grad():
            batch_features = extractor(batch.to(device))
        real_features_list.append(batch_features.detach().cpu().numpy())
    real_features = np.concatenate(real_features_list, axis=0)

    # Extract features for synthetic dataset
    synth_features_list = []
    for batch in tqdm(synth_loader, desc="Processing Synthetic Images -> dino vits8"):
        with torch.no_grad():
            batch_features = extractor(batch.to(device))
        synth_features_list.append(batch_features.detach().cpu().numpy())
    fake_features = np.concatenate(synth_features_list, axis=0)

    ip_result = ImprovedPrecision(k=6).compute(real_features=real_features, fake_features=fake_features)
    ir_result = ImprovedRecall(k=6).compute(real_features=real_features, fake_features=fake_features)
    density_result = Density(k=6).compute(real_features=real_features, fake_features=fake_features)
    coverage_result = Coverage(k=6).compute(real_features=real_features, fake_features=fake_features)

    results_dict = {
        "precision": ip_result.value[0],
        "recall": ir_result.value[0],
        "density": density_result.value[0],
        "coverage": coverage_result.value[0],
    }

    # UMAP 2D visualization
    fig = umap_visualization(real_features, fake_features)

    return results_dict, fig


def calculate_fid_metric(real_dataset, synth_dataset, device, batch_size=64):
    """Calculate the Frechet Inception Distance (FID) between two datasets using the implementation from pymdma library."""
    # inception feature extractor -> used for FID calculation
    extractor = ExtractorFactory.model_from_name(name="inception_fid").to(device)

    # data loader
    real_dataset.set_default_transformation(False)
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    synth_dataset.set_default_transformation(False)
    synth_loader = DataLoader(synth_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    # Extract features for real dataset
    real_features_list = []
    for batch in tqdm(real_loader, desc="Processing Real Images -> inception"):
        with torch.no_grad():
            batch_features = extractor(batch.to(device))
        real_features_list.append(batch_features.detach().cpu().numpy())
    real_features = np.concatenate(real_features_list, axis=0)

    # Extract features for synthetic dataset
    synth_features_list = []
    for batch in tqdm(synth_loader, desc="Processing Synthetic Images -> inception"):
        with torch.no_grad():
            batch_features = extractor(batch.to(device))
        synth_features_list.append(batch_features.detach().cpu().numpy())
    fake_features = np.concatenate(synth_features_list, axis=0)

    fid_result = FrechetDistance().compute(real_features, fake_features)

    return fid_result.value[0]
