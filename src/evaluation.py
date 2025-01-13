"""Evaluation module. Contains methods to evaluate the synthetic images."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

from src.classifier.metrics import (
    compute_confusion_distance,
    compute_entropy,
    compute_norm_entropy,
)
from src.dataset.aux import LABELS


def create_2D_probability_grid(images, probabilities, n_cols):
    """Create a grid of images with probabilities.

    The images are grouped into bins acording to their probabilities. We are assuming that the probabilities are in the same order as the images.
    """
    # Get the size and mode from the first image
    img_size = images[0].size
    img_mode = images[0].mode

    # Group images into bins based on their probabilities
    bins = [[] for _ in range(n_cols)]
    for img, prob in zip(images, probabilities):
        bin_index = min(math.floor(prob * n_cols), n_cols - 1)
        bins[bin_index].append(img)

    # Prepare the grid image
    max_rows = max(len(bin) for bin in bins)
    grid_image = Image.new(img_mode, (n_cols * img_size[0], max_rows * img_size[1]), color="black")

    # Paste images into the grid
    for col, bin_images in enumerate(bins):
        for row, img in enumerate(bin_images):
            grid_image.paste(img, (col * img_size[0], row * img_size[1]))

    return grid_image


def create_metric_grid(images, probs, ordered_results, threshold=0.8, n_cols=10):
    """Create a grid of images with probabilities and labels. We assume that the ordered_results dataframe is sorted by the metric value. The threshold is used to determine the number of classes to display, and can be updated as needed."""
    # start label computation...

    # all columns except the first (image_id) and last (metric)
    labels = ordered_results.columns[1:-1]
    num_samples = len(images)

    # sort probs and calculate the cum sum
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
    cumulative_sum = torch.cumsum(sorted_probs, dim=1)

    # mask to identify the relevant indices, and then count the number of relevant probabilities
    mask = cumulative_sum <= threshold
    top_k = mask.sum(dim=1)

    # create label per image
    image_labels = []
    for i in range(num_samples):
        # predicted class
        label = f"{labels[sorted_indices[i, 0].item()]}: {sorted_probs[i, 0].item():.2f}"
        # other classes, if ambiguous
        for j in range(1, top_k[i]):
            label += f"\n{labels[sorted_indices[i, j].item()]}: {sorted_probs[i, j].item():.2f}"
        image_labels.append([i, label])
    # add labels to the results
    ordered_results = ordered_results.merge(pd.DataFrame(image_labels, columns=["image_id", "label"]), on="image_id")
    # update label with metric value
    ordered_results["label"] = (
        ordered_results["entropy"].apply(lambda x: f"Entropy: {x:.2f}") + "\n" + ordered_results["label"]
    )

    # start grid creation...

    # prepare grid
    n_rows = (num_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.5 * n_rows))
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    # for loop on dataframe
    for i, row in ordered_results.iterrows():
        axes[i].imshow(images[row["image_id"]])
        axes[i].set_title(row["label"], fontsize=8)
        axes[i].axis("off")

    return fig


def curate_results(probs, dataset_name, metric):
    """Curate the results in a dataframe format, where the first column is the image_id and the last is the value for the guidance metric. All the columns in between are the classes. The dataframe is sorted by the metric."""
    # transform the probabilities into a dataframe format
    labels = LABELS[dataset_name]
    results = pd.DataFrame(probs.detach().cpu().numpy(), columns=labels).reset_index()
    results = results.rename(columns={"index": "image_id"})

    # compute metric per image
    value = None
    if metric == "entropy":
        value = compute_entropy(probs).detach().cpu().numpy()
    elif metric == "norm_entropy":
        value = compute_norm_entropy(probs).detach().cpu().numpy()
    elif metric == "acd":
        value = compute_confusion_distance(probs).detach().cpu().numpy()
    results[metric] = value

    results.sort_values(by=metric, ascending=False, inplace=True)
    return results


def sample_synthetic_images(synth_dataset, sample_size, classifier, metric, device, n_cols=10):
    """Visualize the synthetic images in a grid. If a classifier is provided, also take that into account."""
    # sample images for visualization
    # temporary fix for MNIST dataset
    synth_dataset.set_convert_rgb(synth_dataset.get_dataset_name() != "mnist")
    sampled_tensors, sampled_images = synth_dataset.sample_to_tensor(sample_size)

    grid = None
    results = None
    # log the sample grid
    if classifier:
        # probabilities for the sampled images
        sampled_probs = classifier.predict(sampled_tensors.to(device))
        results = curate_results(sampled_probs, synth_dataset.get_dataset_name(), metric)

        # specific grid for binary classification -> same as GASTeN
        if sampled_probs.size(1) == 2:
            grid = create_2D_probability_grid(sampled_images, sampled_probs, n_cols)
        else:
            # threshold (0.8) may be adjusted
            grid = create_metric_grid(sampled_images, sampled_probs, results, 0.8, n_cols)
    else:
        # if no classifier provided, display a generic grid
        n_rows = math.ceil(len(sampled_images) / n_cols)
        grid = make_image_grid(sampled_images, rows=n_rows, cols=n_cols)
    return grid, results


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
