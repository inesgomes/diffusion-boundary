"""Visualization module. Contains methods to visualize distributions and images."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from umap import UMAP

from src.evaluation import compute_classes_confusion


def format_label(row, n_classes: int, metrics: list):
    """Format the legend to be added on the display of a given image.

    Always shows top 3 classes and their respective probability value; KDN and their respective neighbors; It is also possible to enrich with other metrics if provided.
    """
    # if label name is longer than 20 characters, truncate it
    top_labels = row["sorted_labels"][:n_classes]
    probs_legend = "\n".join(
        [
            f"{label_name if len(label_name) <= 20 else label_name[:20] + '...'}: {row[label_name]:.2f}"
            for label_name in top_labels
        ]
    )
    # kdn_legend = f"KDN: {row['kdn']:.2f}\n nbr: {row['kdn_nbr']}"
    metrics_legend = "\n".join([f"{m}: {row[m]:.2f}" for m in metrics])

    # return kdn_legend + "\n" + metrics_legend + "\n" + probs_legend
    return metrics_legend + "\n" + probs_legend


def visualize_2D_probability_grid(images, probabilities, n_cols):
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


def visualize_top_synthetic_metric(images_dataset, results, sort_metric, ascending, top_n=5, display_rgb=True):
    """Visualize the top N synthetic images based on a given metric."""
    # sort results by metric and select top N
    top_results = results.sort_values(by=sort_metric, ascending=ascending).head(top_n)
    # create figure
    fig, axes = plt.subplots(1, top_n, figsize=(1.5 * top_n, 2.5))

    images_dataset.set_use_transformation("NONE")
    for i, (_, row) in enumerate(top_results.iterrows()):
        img, _ = images_dataset[row["image_id"]]
        if not display_rgb:
            axes[i].imshow(img.convert("L"), cmap="gray")
        else:
            axes[i].imshow(img)
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

    # check how many elements have less than half of the max value for the metric (minimize version)
    mask = synth_dataset_res[sort_metric] <= synth_dataset_res[sort_metric].max() / 2
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
        img, _ = synth_dataset[row["image_id"]]
        if not display_rgb:
            axes[i].imshow(img.convert("L"), cmap="gray")
        else:
            axes[i].imshow(img)
        # calculate label
        label = format_label(row, 5, [sort_metric])
        axes[i].set_title(label, fontsize=8)
        axes[i].axis("off")

    synth_dataset.set_use_transformation("DEFAULT")

    return fig, results


def visualize_metrics_distributions(real_synth_results, metrics):
    """Plot distributions for real and synthetic datasets."""
    # boxplot real vs synthetic dataset -> metrics
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


def visualize_class_distributions(real_synth_results, classes, n_classes):
    """Plot distributions for real and synthetic datasets per each selected class vs the sum of all classes."""
    # sum all probabilities in one class
    new_classes = classes[:]
    real_synth_results["other"] = (
        real_synth_results.drop(columns=new_classes).iloc[:, 2 : n_classes + 2 - len(new_classes)].sum(axis=1)
    )
    new_classes.append("other")

    # melt classes + other
    viz_results_melt = real_synth_results.melt(
        id_vars="keys", value_vars=new_classes, var_name="label", value_name="probability"
    )

    fig_classes, ax_c = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    sns.kdeplot(
        data=viz_results_melt[viz_results_melt["keys"] == "Real"],
        x="probability",
        hue="label",
        ax=ax_c[0],
    )
    ax_c[0].set_title("Real")
    ax_c[0].set_xlim(-0.1, 1.1)

    sns.kdeplot(
        data=viz_results_melt[viz_results_melt["keys"] == "Synthetic"],
        x="probability",
        hue="label",
        ax=ax_c[1],
    )
    ax_c[1].set_title("Synthetic")
    ax_c[1].set_xlim(-0.1, 1.1)

    fig_classes.suptitle("Class Probabilities Distribution")
    return fig_classes


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


def visualize_features_umap(real_features, real_labels, synth_features, guidance_metric):
    """2D UMAP visualization of the features. Returns the figure."""
    umap = UMAP(n_components=2, random_state=10)
    real_feats_2d = umap.fit_transform(real_features)
    fake_feats_2d = umap.transform(synth_features)

    marker_styles = ["o", "s", "^", "D", "v", "P", "*", "X", "H", "+"]

    fig, ax = plt.subplots(figsize=(10, 10))

    unique_labels = np.unique(real_labels)
    for i, label in enumerate(unique_labels):
        marker = marker_styles[i % len(marker_styles)]
        indices = real_labels == label
        ax.scatter(
            real_feats_2d[indices, 0],
            real_feats_2d[indices, 1],
            marker=marker,
            label=label,
            edgecolors="black",  # optional for contrast
            facecolors="none",  # optional for clarity
        )

    cmap = sns.cubehelix_palette(as_cmap=True)
    points = ax.scatter(
        fake_feats_2d[:, 0], fake_feats_2d[:, 1], c=guidance_metric, label="synthetic", marker=".", alpha=0.8, cmap=cmap
    )
    fig.colorbar(points)

    # plt.scatter(real_feats_2d[:, 0], real_feats_2d[:, 1], s=3, label=f"Real (n={real_feats_2d.shape[0]})", color="red")
    # plt.scatter(fake_feats_2d[:, 0], fake_feats_2d[:, 1], s=3, label=f"Fake (n={fake_feats_2d.shape[0]})", color="blue")
    plt.title("UMAP Features Visualization | Real vs Synthetic")
    plt.legend()
    return fig
