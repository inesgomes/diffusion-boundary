"""Metrics for evaluating the classifier output. Can be used for guidance and for visualization purposes."""

import torch


def compute_entropy(probs):
    """Calculate the entropy of the classifier output."""
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)


def compute_confusion_distance(probs):
    """
    Calculate the confusion distance of the classifier output.

    (CD) = |0.5 - probs|
    """
    return (0.5 - probs).abs()


def compute_norm_entropy(probs):
    """Calculate the normalized entropy of the classifier output."""
    entropy = compute_entropy(probs)
    return entropy / torch.log(torch.tensor(probs.size(1)).float())


def compute_metric(metric, probs):
    """Calculate the specified metric of the classifier output."""
    if metric == "entropy":
        return compute_entropy(probs)
    if metric == "confusion_distance":
        return compute_confusion_distance(probs)
    if metric == "norm_entropy":
        return compute_norm_entropy(probs)
    raise ValueError(f"Metric {metric} not supported")
