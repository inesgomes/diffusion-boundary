"""Metrics for evaluating the classifier output. Can be used for guidance and for visualization purposes."""

import torch

MULTICLASS_METRICS = ["entropy", "norm_entropy", "margin"]
BINARY_METRICS = ["confusion_distance", "margin"]


def compute_confusion_distance(probs):
    """
    Calculate the confusion distance of the classifier output. Only for binary classification.

    (CD) = |0.5 - probs|
    """
    return (0.5 - probs).abs()


def compute_entropy(probs):
    """Calculate the entropy of the classifier output."""
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)


def compute_norm_entropy(probs):
    """Calculate the normalized entropy of the classifier output."""
    entropy = compute_entropy(probs)
    return entropy / torch.log(torch.tensor(probs.size(1)).float())


def compute_margin(probs):
    """Calculate the margin of the classifier output. Per each data point, find the difference between the highest and the lowest probability.

    # TODO confirm this is working
    """
    return probs.max(dim=1).values - probs.min(dim=1).values


def compute_metric(metric, probs):
    """Calculate the specified metric of the classifier output."""
    if metric == "entropy":
        return compute_entropy(probs)
    if metric == "confusion_distance":
        return compute_confusion_distance(probs)
    if metric == "norm_entropy":
        return compute_norm_entropy(probs)
    if metric == "margin":
        return compute_margin(probs)
    raise ValueError(f"Metric {metric} not supported")
