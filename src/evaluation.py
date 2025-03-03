"""Evaluation module. Contains methods to evaluate the synthetic images."""

import math
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from umap import UMAP

from src.classifier.metrics import (
    BINARY_METRICS,
    MULTICLASS_METRICS,
    UNCERTAINTY_METRICS,
    compute_metric,
)


def create_2D_probability_grid(images, probabilities, n_cols):
    """Create a grid of images with probabilities.

    The images are grouped into bins acording to their probabilities. We are assuming that the probabilities are in the same order as the images.

    TODO: this is currently not being used.
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


def format_label(row, n_classes, metric):
    """Format the legend to be added on the display of a given image."""
    top_labels = row["sorted_labels"][:n_classes]
    legend = "\n".join([f"{label_name}: {row[label_name]:.2f}" for label_name in top_labels])
    return f"{metric}: {row[metric]:.2f}\n" + legend


def visualize_top_synthetic_metric(images_dataset, results, sort_metric, display_rgb=True):
    """Visualize the top 5 synthetic images based on a given metric."""
    N_SAMPLES = 5
    top_results = results.sort_values(by=sort_metric, ascending=False).head(N_SAMPLES)

    fig, axes = plt.subplots(1, N_SAMPLES, figsize=(1.5 * N_SAMPLES, 2.5))

    images_dataset.set_use_transformation("NONE")
    for i, (_, row) in enumerate(top_results.iterrows()):
        if not display_rgb:
            axes[i].imshow(images_dataset[row["image_id"]].convert("L"), cmap="gray")
        else:
            axes[i].imshow(images_dataset[row["image_id"]])
        label = f"{row[sort_metric]:.2f}"
        axes[i].set_title(label, fontsize=8)
        axes[i].axis("off")
    images_dataset.set_use_transformation("DEFAULT")

    return fig


def visualize_sample_synthetic_images(
    synth_dataset, synth_dataset_res, sample_size, sort_metric, display_rgb, n_cols=10
):
    """Visualize the synthetic images in a grid. If a classifier is provided, also take that into account."""
    # order dataset by metric and select top sample
    results = synth_dataset_res.sort_values(by=sort_metric, ascending=False).head(sample_size)

    # check how many elements have at least half of the max value for the metric
    mask = synth_dataset_res[sort_metric] >= synth_dataset_res[sort_metric].max() / 2
    if mask.sum() <= sample_size:
        results = synth_dataset_res.sort_values(by=sort_metric, ascending=False).head(sample_size)
    else:
        results = synth_dataset_res[mask].sample(sample_size)

    # prepare grid
    n_rows = (results.shape[0] + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    synth_dataset.set_use_transformation("NONE")

    for i, (_, row) in enumerate(results.iterrows()):
        if not display_rgb:
            axes[i].imshow(synth_dataset[row["image_id"]].convert("L"), cmap="gray")
        else:
            axes[i].imshow(synth_dataset[row["image_id"]])
        # calculate label
        label = format_label(row, 3, sort_metric)
        axes[i].set_title(label, fontsize=8)
        axes[i].axis("off")

    synth_dataset.set_use_transformation("DEFAULT")

    return fig, results


def calculate_performance_metrics(probs, gt):
    """Calculate the performance of the model."""
    # TODO: we may add here other performance metrics
    predictions = probs.argmax(dim=1).detach().cpu().numpy()
    acc = accuracy_score(predictions, gt)
    print(f"Accuracy: {acc:.2%}")


def curate_results(class_labels, probs, probs_dropout=None):
    """Curate the results in a dataframe format.

    The first column is the image_id and the last is the value for the guidance metric.
    All the columns in between are the classes. The dataframe is sorted by the metric.
    """
    # transform the probabilities into a dataframe format
    results = pd.DataFrame(probs.detach().cpu().numpy(), columns=class_labels).reset_index()

    # add an id per each image
    results.rename(columns={"index": "image_id"}, inplace=True)

    # sort the probabilities and their corresponding labels in descending order
    sorted_indices = np.argsort(-results[class_labels].values, axis=1)
    class_labels_array = np.array(class_labels)
    sorted_labels = class_labels_array[sorted_indices]
    results["sorted_labels"] = [list(row) for row in sorted_labels]
    results["pred"] = sorted_labels[:, 0]

    # compute all extra metrics per image and add to the dataframe
    metrics = BINARY_METRICS if probs.size(1) == 2 else MULTICLASS_METRICS
    metrics = list(set(metrics) | set(UNCERTAINTY_METRICS))
    for metric in metrics:
        results[metric] = compute_metric(metric, probs, probs_dropout=probs_dropout).detach().cpu().numpy()

    return results


def prepare_dataset_results(dataset, classifier, batch_size, device, num_samples=None, drop_threshold=None, gt=None):
    """Prepare the dataset results for visualization. First compute the predictions (in batch), then apply the curate_results function."""
    # compute dataset predictions
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    probs_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Compute predictions"):
            batch = batch.to(device)
            probs_batch, _ = classifier.predict(batch)
            probs_list.append(probs_batch.cpu())
        probs = torch.cat(probs_list, dim=0)
    # calculate accuracy, if labels exist
    if gt is not None:
        calculate_performance_metrics(probs, gt)

    # compute predictions with MC dropout method
    probs_dropout = None
    if (num_samples is not None) and (drop_threshold is not None):
        # first, set dropout and change model to training mode
        classifier.set_dropout(dropout_p=drop_threshold)
        classifier.set_train()

        all_predictions = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Compute MC dropout predictions"):
                batch = batch.to(device)
                batch_predictions = []
                for _ in range(num_samples):
                    probs_batch, _ = classifier.predict(batch)
                    batch_predictions.append(probs_batch.unsqueeze(0))
                batch_predictions = torch.cat(batch_predictions, dim=0)
                all_predictions.append(batch_predictions)
        probs_dropout = torch.cat(all_predictions, dim=1).cpu()

        # return to eval mode
        classifier.set_dropout(dropout_p=0)
        classifier.set_eval()

    # prepare dataframe with the probabilities + metrics per image
    return curate_results(dataset.get_class_labels(), probs, probs_dropout)


def visualize_metrics_distributions(real_synth_results, n_classes):
    """Plot distributions for real and synthetic datasets."""
    # boxplot real vs synthetic dataset -> metrics
    metrics = MULTICLASS_METRICS if n_classes >= 2 else BINARY_METRICS
    metrics = list(set(metrics) | set(UNCERTAINTY_METRICS))
    fig_metric, ax_m = plt.subplots(figsize=(12, 1.2 * len(metrics)))

    viz_metrics_melt = real_synth_results.melt(
        id_vars="keys", value_vars=metrics, var_name="metric", value_name="value"
    )

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

    return fig_metric


def visualize_class_distributions(real_synth_results, top_n):
    """Plot distributions for real and synthetic datasets per each class."""
    # top classes
    top_5_classes = real_synth_results.groupby("pred").size().sort_values(ascending=False).head(top_n)
    print(top_5_classes)

    real_synth_results_filter = real_synth_results[real_synth_results["pred"].isin(top_5_classes.index)]

    # probs per class
    fig_classes, ax_c = plt.subplots(figsize=(8, 1.5 * top_n))

    viz_results_melt = real_synth_results_filter.melt(
        id_vars="keys", value_vars=top_5_classes.index, var_name="label", value_name="probability"
    )
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

    return fig_classes


def compute_classes_confusion(results, threshold):
    """For a given dataset, given the threshold of certainty defined, compute which classes are most likely to be ambiguous."""
    # given the probs, we cumulatively sum, and then apply the threshold to get the classes list

    # compute the ambiguous classes per image
    selected_classes = []
    pairs_classes = []
    for _, row in results.iterrows():
        # sort the probabilities and their corresponding labels in descending order
        # TODO: this can be improved by using the sorted_labels column
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
    matrix_real, lst_real = compute_classes_confusion(real_results[labels], threshold)
    matrix_synth, lst_synth = compute_classes_confusion(synth_results[labels], threshold)

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

    # dataloader
    real_dataset.set_use_transformation("NORM")
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    synth_dataset.set_use_transformation("NORM")
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

    real_dataset.set_use_transformation("DEFAULT")
    synth_dataset.set_use_transformation("DEFAULT")

    return fid_result.value[0]
