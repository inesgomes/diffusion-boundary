"""Evaluation module. Contains methods to evaluate the synthetic images."""

from itertools import combinations

import numpy as np
import pandas as pd
import torch
from pymdma.image.measures.synthesis_val import (
    Coverage,
    Density,
    FrechetDistance,
    ImprovedPrecision,
    ImprovedRecall,
)
from pymdma.image.models.features import ExtractorFactory
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classifier.metrics import (
    BINARY_METRICS,
    MULTICLASS_METRICS,
    UNCERTAINTY_METRICS,
    compute_metric,
)

EVAL_METRICS = ["kdn"]


def compute_probabilities(classifier, dataset, device):
    """Compute the probabilities for a given dataset.

    The function will compute the probabilities for each sample in the dataset, and return a tensor with the probabilities.
    The tensor will have the shape (num_samples, num_classes, num_images).
    """
    probs_list = []
    labels_list = []
    with torch.no_grad():
        for batch_images, batch_labels in tqdm(dataset, desc="Compute predictions"):
            batch_images = batch_images.to(device)
            probs_batch, _ = classifier.predict(batch_images)
            probs_list.append(probs_batch.cpu())
            labels_list.append(batch_labels)
        probs = torch.cat(probs_list, dim=0)
        labels = np.concatenate(labels_list, axis=0)

    # if all values are -1, return None on the labels
    all_neg_ones = (labels == -1).all()
    final_labels = labels if not all_neg_ones else None

    return probs, final_labels


def compute_mc_droupout_metrics(classifier, dataset, num_samples, drop_threshold, device):
    """Compute the MC dropout metrics for a given dataset.

    The function will compute the probabilities for each sample in the dataset, and return a tensor with the probabilities.
    The tensor will have the shape (num_samples, num_classes, num_images).
    """
    probs_dropout = None
    if (num_samples is not None) and (drop_threshold is not None):
        # first, set dropout and change model to training mode
        classifier.set_dropout(dropout_p=drop_threshold)
        classifier.set_train()

        all_predictions = []
        with torch.no_grad():
            for batch, _ in tqdm(dataset, desc="Compute MC dropout predictions"):
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
    return probs_dropout


def print_performace_metrics(probs, labels):
    """Calculate performance metrics for the classifier."""
    predictions = probs.argmax(dim=1).detach().cpu().numpy()
    unique_labels = np.unique(labels)

    # accuracy
    acc = accuracy_score(predictions, labels)
    print(f"Accuracy: {acc:.2%}")

    # macro F1-score
    macro_f1 = f1_score(labels, predictions, average="macro", labels=unique_labels)
    print(f"Macro F1-score: {macro_f1:.2f}")

    # precision, recall, f1 per class
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=None, labels=unique_labels)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)


def prepare_dataset_results(dataset, classifier, target, batch_size, device, num_samples=None, drop_threshold=None):
    """Prepare the dataset results for visualization.

    First compute the predictions (in batch). Then, curate the results in a dataframe format.
    The first column is the image_id and the last is the value for the guidance metric.
    All the columns in between are the classes. The dataframe is sorted by the metric.
    """
    # prepare the dataset
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    # compute dataset predictions
    probs, labels = compute_probabilities(classifier, loader, device)

    # compute predictions with MC dropout method, if asked
    probs_dropout = compute_mc_droupout_metrics(classifier, loader, num_samples, drop_threshold, device)

    # transform the probabilities into a dataframe format
    class_labels = dataset.get_class_labels()
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
    target_idx = [dataset.get_class_idx(class_name) for class_name in target]
    for metric in metrics:
        metric_result = compute_metric(metric, probs, probs_dropout=probs_dropout, labels_idx=target_idx)
        if not torch.all(torch.isnan(metric_result)):
            results[metric] = torch.abs(metric_result).detach().cpu().numpy()

    # (extra) calculate performance metrics and show it, if labels exist
    if labels is not None:
        print_performace_metrics(probs, labels)

    return results


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


def compute_kdn(real_features, real_labels, fake_features, boundary_classes, k=6):
    """Calculate the K-Disagreeing Neighbors score (KDN) of each synthtic sample, comparing to the real dataset."""
    # fit on real features
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree", metric="euclidean")
    nbrs.fit(real_features)

    # find k-nearest neighbors for each synthetic feature
    _, indices = nbrs.kneighbors(fake_features)

    disagreements = []
    nbr_labels_lst = []
    # iterate per each synthetic samples
    for neighbors in indices:
        neighbor_labels = real_labels[neighbors]
        nbr_labels_lst.append(neighbor_labels)

        # compute proportions of neighbors from each class
        counts = {b_cls: np.sum(neighbor_labels == b_cls) for b_cls in boundary_classes}
        proportions = np.array([counts[b_cls] / k for b_cls in boundary_classes])

        # if all neighbors belong to the same class, disagreement = 0
        if np.max(proportions) == 1.0:
            disagreement = 0
        else:
            # maximum disagreement when classes (from boundary) are equally split
            ideal_split = 1 / len(boundary_classes)
            disagreement = 1 - np.sum(np.abs(proportions - ideal_split))

        disagreements.append(disagreement)

    # indices as a list of strings
    # TODO update to get the real label name
    lst_nbr = [" ".join(map(str, sublist)) for sublist in nbr_labels_lst]

    return disagreements, lst_nbr


def calculate_feature_metrics(real_dataset, fake_dataset, target, batch_size, device):
    """Calculate evaluation metrics and a visualization needed based on features extracted from the images."""
    # label to idx
    boundary_labels = [real_dataset.get_class_idx(class_name) for class_name in target]

    # compute features from each image (real and synthetic)
    real_features, real_labels = get_dataset_features(real_dataset, batch_size, device)
    fake_features, _ = get_dataset_features(fake_dataset, batch_size, device)

    # calculate metrics
    kdn_results, kdn_nbr = compute_kdn(real_features, real_labels, fake_features, boundary_labels, k=10)
    synth_metrics = pd.DataFrame({"kdn": kdn_results, "kdn_nbr": kdn_nbr})

    return synth_metrics, real_features, fake_features


def get_dataset_features(dataset, batch_size, device):
    """Extract features using dino_vits8 from a given image dataset. Also returns the dataset labels."""
    # extract features to compute quality metrics
    extractor = ExtractorFactory.model_from_name(name="dino_vits8").to(device)

    # data loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    # Extract features and ground truth for real dataset
    features_list = []
    labels_list = []
    with torch.no_grad():
        for batch_images, batch_labels in tqdm(loader, desc="Processing Real Images -> dino vits8"):
            batch_images = batch_images.to(device)
            batch_features = extractor(batch_images)
            features_list.append(batch_features.detach().cpu().numpy())
            labels_list.append(batch_labels)
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return features, labels


def calculate_fid_metric(real_dataset, synth_dataset, batch_size, device):
    """Calculate the Frechet Inception Distance (FID) between two datasets using the implementation from pymdma library."""
    # inception feature extractor -> used for FID calculation
    extractor = ExtractorFactory.model_from_name(name="inception_fid").to(device)

    # dataloader with specific transformation
    real_dataset.set_use_transformation("NORM")
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    synth_dataset.set_use_transformation("NORM")
    synth_loader = DataLoader(synth_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    # Extract features for real dataset
    real_features_list = []
    with torch.no_grad():
        for batch, _ in tqdm(real_loader, desc="Processing Real Images -> inception"):
            batch = batch.to(device)
            batch_features = extractor(batch)
            real_features_list.append(batch_features.detach().cpu().numpy())
    real_features = np.concatenate(real_features_list, axis=0)

    # Extract features for synthetic dataset
    synth_features_list = []
    with torch.no_grad():
        for batch, _ in tqdm(synth_loader, desc="Processing Synthetic Images -> inception"):
            batch = batch.to(device)
            batch_features = extractor(batch)
            synth_features_list.append(batch_features.detach().cpu().numpy())
    fake_features = np.concatenate(synth_features_list, axis=0)

    fid_result = FrechetDistance().compute(real_features, fake_features)

    real_dataset.set_use_transformation("DEFAULT")
    synth_dataset.set_use_transformation("DEFAULT")

    return fid_result.value[0]


def calculate_evaluation_metrics(real_features, fake_features, synthetic_data_res, metrics):
    """Calculate the dataset evaluation metrics.

    Include metrics of quality (improved precision, improved recall, coverage and density), as well as the mean and median of all metrics computed per each sample.
    """
    # quality metrics
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

    # add average and median of selected metrics
    # mean_values = synthetic_data_res[metrics].mean()
    # sd_values = synthetic_data_res[metrics].std()
    median_values = synthetic_data_res[metrics].median()
    q1_values = synthetic_data_res[metrics].quantile(0.25)
    q3_values = synthetic_data_res[metrics].quantile(0.75)

    for m in metrics:
        # results_dict[f"{m}_avg"] = mean_values[m]
        # results_dict[f"{m}_std"] = sd_values[m]
        results_dict[f"{m}_median"] = median_values[m]
        results_dict[f"{m}_iqr"] = q3_values[m] - q1_values[m]
        results_dict[f"{m}_25"] = q1_values[m]
        results_dict[f"{m}_75"] = q3_values[m]

    return results_dict
