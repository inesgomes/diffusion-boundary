"""Evaluation module. Contains methods to evaluate the synthetic images."""

import math
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from umap import UMAP

from src.classifier.metrics import BINARY_METRICS, MULTICLASS_METRICS, compute_metric
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


def create_metric_grid(images, probs, results, sort_metric, display_rgb, threshold, n_cols=10):
    """Create a grid of images with probabilities and labels. Only for multiclassification.

    The threshold is used to determine the number of classes to display, and can be updated as needed.
    If there is no metric defined, we will asume entropy.
    """
    # check if metric is defined
    metric = "entropy" if sort_metric is None else sort_metric

    # sort the results by the metric
    ordered_results = results.sort_values(by=metric, ascending=False)

    # start label computation...

    # first column is (image_id) and next ones are the metrics
    labels = ordered_results.columns[1 : 1 + probs.size(1)]
    num_samples = len(images)

    # sort probs and calculate the cum sum
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
    cumulative_sum = torch.cumsum(sorted_probs, dim=1)

    # mask to identify the relevant indices, and then count the number of relevant probabilities
    mask = cumulative_sum <= threshold
    top_k = mask.sum(dim=1) + 1

    # create label per image
    image_labels = []
    for i in range(num_samples):
        # predicted class
        label = ""
        # other classes, if ambiguous
        for j in range(top_k[i]):
            label += f"{labels[sorted_indices[i, j].item()]}: {sorted_probs[i, j].item():.2f}\n"
        image_labels.append([i, label])
    # add labels to the results
    ordered_results = ordered_results.merge(pd.DataFrame(image_labels, columns=["image_id", "label"]), on="image_id")
    # update label with metric value
    ordered_results["label"] = (
        ordered_results[metric].apply(lambda x: f"{metric}: {x:.2f}\n") + ordered_results["label"]
    )

    # start grid creation...

    # prepare grid
    n_rows = (num_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 2.5 * n_rows))
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    # for loop on dataframe
    for i, row in ordered_results.iterrows():
        if not display_rgb:
            axes[i].imshow(images[row["image_id"]].convert("L"), cmap="gray")
        else:
            axes[i].imshow(images[row["image_id"]])
        axes[i].set_title(row["label"], fontsize=8)
        axes[i].axis("off")

    return fig


def curate_results(probs, dataset_name, n_classes):
    """Curate the results in a dataframe format.

    The first column is the image_id and the last is the value for the guidance metric.
    All the columns in between are the classes. The dataframe is sorted by the metric.
    """
    # transform the probabilities into a dataframe format
    labels = LABELS[dataset_name]
    results = pd.DataFrame(probs.detach().cpu().numpy(), columns=labels).reset_index()
    results = results.rename(columns={"index": "image_id"})

    # compute all extra metrics per image and add to the dataframe
    metrics = BINARY_METRICS if n_classes == 2 else MULTICLASS_METRICS
    for metric in metrics:
        results[metric] = compute_metric(metric, probs).detach().cpu().numpy()
    return results


def visualize_sample_synthetic_images(
    synth_dataset, sample_size, classifier, sort_metric, display_rgb, certainty_threshold, device, n_cols=10
):
    """Visualize the synthetic images in a grid. If a classifier is provided, also take that into account."""
    # sample images for visualization
    sampled_tensors, sampled_images = synth_dataset.sample_to_tensor(sample_size)

    grid = None
    results = None
    # log the sample grid
    if classifier:
        # probabilities for the sampled images
        with torch.no_grad():
            sampled_tensors = sampled_tensors.to(device)
            sampled_probs = classifier.predict(sampled_tensors)
        results = curate_results(sampled_probs, synth_dataset.get_dataset_name(), synth_dataset.get_n_classes())

        # specific grid for binary classification -> same as GASTeN
        if sampled_probs.size(1) == 2:
            grid = create_2D_probability_grid(sampled_images, sampled_probs, n_cols)
        else:
            grid = create_metric_grid(
                sampled_images, sampled_probs, results, sort_metric, display_rgb, certainty_threshold, n_cols
            )
    else:
        # if no classifier provided, display a generic grid
        n_rows = math.ceil(len(sampled_images) / n_cols)
        grid = make_image_grid(sampled_images, rows=n_rows, cols=n_cols)
    return grid, results


def prepare_dataset_results(dataset, classifier, batch_size, device, gt=None):
    """Prepare the dataset results for visualization. First compute the predictions (in batch), then apply the curate_results function."""
    # compute dataset predictions
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    probs_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Compute predictions"):
            batch = batch.to(device)
            probs_batch = classifier.predict(batch)
            probs_list.append(probs_batch.cpu())
        probs = torch.cat(probs_list, dim=0)

    if gt is not None:
        predictions = probs.argmax(dim=1).detach().cpu().numpy()
        acc = accuracy_score(predictions, gt)
        print(f"Accuracy: {acc:.2%}")

    return curate_results(probs, dataset.get_dataset_name(), dataset.get_n_classes())


def visualize_distributions(real_results, synth_results, n_classes):
    """Plot distributions for real and synthetic datasets."""
    viz_results = pd.concat([real_results, synth_results], keys=["Real", "Synthetic"]).reset_index()
    viz_results = viz_results.rename(columns={"level_0": "keys"}).drop(columns=["level_1"])

    # boxplot real vs synthetic dataset -> metrics
    metrics = viz_results.columns[n_classes + 2 :]
    fig_metric, ax_m = plt.subplots(figsize=(8, 1.5 * len(metrics)))

    viz_metrics_melt = viz_results.melt(id_vars="keys", value_vars=metrics, var_name="metric", value_name="value")

    sns.boxplot(
        data=viz_metrics_melt,
        x="value",
        y="metric",
        hue="keys",
        ax=ax_m,
        orient="h",
        hue_order=["Real", "Synthetic"],
        palette=["red", "blue"],
        gap=0.1,
        width=0.5,
        fill=False,
        fliersize=1,
    )
    ax_m.set_title("Metric values Distribution | Real vs Synthetic")

    # probs per class
    labels = viz_results.columns[2 : n_classes + 2]
    fig_classes, ax_c = plt.subplots(figsize=(8, 1.5 * len(labels)))

    viz_results_melt = viz_results.melt(id_vars="keys", value_vars=labels, var_name="label", value_name="probability")
    sns.boxplot(
        data=viz_results_melt,
        x="probability",
        y="label",
        hue="keys",
        ax=ax_c,
        orient="h",
        hue_order=["Real", "Synthetic"],
        palette=["red", "blue"],
        gap=0.1,
        width=0.5,
        fill=False,
        fliersize=1,
    )
    ax_c.set_title("Class Probabilities Distribution | Real vs Synthetic")

    return fig_metric, fig_classes


def compute_classes_confusion_confusion(results, threshold):
    """For a given dataset, given the threshold of certainty defined, compute which classes are most likely to be ambiguous."""
    # given the probs, we cumulatively sum, and then apply the threshold to get the classes list

    # compute the ambiguous classes per image
    selected_classes = []
    pairs_classes = []
    for _, row in results.iterrows():
        # sort the probabilities and their corresponding labels in descending order
        sorted_probs_labels = sorted(zip(row.values, row.index), reverse=True, key=lambda x: x[0])
        sorted_probs, sorted_labels = zip(*sorted_probs_labels)

        # compute the cumulative sum of the sorted probabilities, and select the classes
        cumsum = pd.Series(sorted_probs).cumsum()
        top_k = (cumsum <= threshold).sum() + 1
        selected = [sorted_labels[i] for i in range(top_k)]
        selected_classes.append(selected)

        # transform the list in subsets of 2, e.g. [A, B, C] = [A, B], [A, C], [B, C]; [A] = [A, A]
        if len(selected) < 2:
            # Handle cases where there are fewer than 2 elements
            pairs_classes.append([selected[0], selected[0]])
        else:
            # Generate all combinations of size 2
            pairs_classes.extend([list(pair) for pair in combinations(selected, 2)])

    # sort to avoid duplicates
    sorted_pairs = [tuple(sorted(pair, reverse=True)) for pair in pairs_classes]
    sorted_selected = [tuple(sorted(tup, reverse=True)) for tup in selected_classes]

    # compute the confusion matrix for the pairs
    df_pairs = pd.DataFrame(sorted_pairs, columns=["class1", "class2"])
    df_pairs["count"] = 1
    df_pairs = df_pairs.groupby(["class1", "class2"], as_index=False)["count"].sum()

    matrix = df_pairs.pivot(index="class1", columns="class2", values="count").fillna(0)
    return matrix, sorted_selected


def visualize_confusion(real_results, synth_results, n_classes, threshold):
    """Visualize the confusion matrix for the real and synthetic datasets."""
    # remove irrelevant columns
    labels = real_results.columns[1 : n_classes + 1]

    # prepare matrix and list of classes
    matrix_real, lst_real = compute_classes_confusion_confusion(real_results[labels], threshold)
    matrix_synth, lst_synth = compute_classes_confusion_confusion(synth_results[labels], threshold)

    # create hetmaps for pairs
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # equal v_max for both heatmaps
    v_max = max(matrix_real.values.max(), matrix_synth.values.max())

    # real heatmap
    mask_real = np.triu(np.ones_like(matrix_real, dtype=bool))
    np.fill_diagonal(mask_real, False)
    sns.heatmap(
        matrix_real,
        ax=axes[0],
        cmap="Reds",
        annot=True,
        fmt="g",
        mask=mask_real,
        vmin=0,
        vmax=v_max,
        cbar=False,
        square=True,
    )
    axes[0].set_title("Real Dataset")

    # synthetic heatmap
    mask_synth = np.triu(np.ones_like(matrix_synth, dtype=bool))
    np.fill_diagonal(mask_synth, False)
    sns.heatmap(
        matrix_synth, ax=axes[1], cmap="Blues", annot=True, fmt="g", mask=mask_synth, vmin=0, vmax=v_max, square=True
    )
    axes[1].set_title("Synthetic Dataset")

    # dataframe of comparison of boundaries between real and synthetic

    # count the number of times each tuple appears in the real and synthetic datasets
    combined_df = pd.DataFrame(
        {"real": pd.Series(lst_real).value_counts(), "synthetic": pd.Series(lst_synth).value_counts()}
    ).fillna(0)
    # compute the difference
    combined_df["difference"] = combined_df["synthetic"] - combined_df["real"]
    # count number of tuples
    combined_df["n_boundaries"] = combined_df.index.map(len)

    # sort by number of boundaries
    combined_df.sort_values(by="n_boundaries", inplace=True)

    return fig, combined_df


def umap_visualization(real_features, synth_features):
    """2D UMAP visualization of the features. Returns the figure."""
    umap = UMAP(n_components=2, random_state=10, n_jobs=1)
    real_feats_2d = umap.fit_transform(real_features)
    fake_feats_2d = umap.transform(synth_features)

    fig = plt.figure(figsize=(10, 10))
    plt.scatter(real_feats_2d[:, 0], real_feats_2d[:, 1], s=3, label=f"Real (n={real_feats_2d.shape[0]})", color="red")
    plt.scatter(fake_feats_2d[:, 0], fake_feats_2d[:, 1], s=3, label=f"Fake (n={fake_feats_2d.shape[0]})", color="blue")
    plt.title("UMAP Features Visualization | Real vs Synthetic")
    plt.legend()
    return fig


def calculate_synthetic_metrics(real_dataset, synth_dataset, batch_size, device):
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
    with torch.no_grad():
        for batch in tqdm(real_loader, desc="Processing Real Images -> dino vits8"):
            batch = batch.to(device)
            batch_features = extractor(batch)
            real_features_list.append(batch_features.detach().cpu().numpy())
    real_features = np.concatenate(real_features_list, axis=0)

    # Extract features for synthetic dataset
    synth_features_list = []
    with torch.no_grad():
        for batch in tqdm(synth_loader, desc="Processing Synthetic Images -> dino vits8"):
            batch = batch.to(device)
            batch_features = extractor(batch)
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


def calculate_fid_metric(real_dataset, synth_dataset, batch_size, device):
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
    with torch.no_grad():
        for batch in tqdm(real_loader, desc="Processing Real Images -> inception"):
            batch_features = extractor(batch.to(device))
            real_features_list.append(batch_features.detach().cpu().numpy())
    real_features = np.concatenate(real_features_list, axis=0)

    # Extract features for synthetic dataset
    synth_features_list = []
    with torch.no_grad():
        for batch in tqdm(synth_loader, desc="Processing Synthetic Images -> inception"):
            batch = batch.to(device)
            batch_features = extractor(batch)
            synth_features_list.append(batch_features.detach().cpu().numpy())
    fake_features = np.concatenate(synth_features_list, axis=0)

    fid_result = FrechetDistance().compute(real_features, fake_features)

    return fid_result.value[0]
