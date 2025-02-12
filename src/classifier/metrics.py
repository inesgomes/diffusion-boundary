"""
Metrics for evaluating the classifier output. Can be used for guidance and for visualization purposes.

We receive the classifier output as a tensor of probabilities. The metrics are calculated per each data point in the batch.
IMPORTANT: all metrics should be implement so that the higher the value, the closer to the decision boundary the classifier is. In the other words, the less confident the classifier is, the higher the metric value should be.
It would also be helpful to have a function that normalizes the metric values to the range [0, 1] for better comparison between different metrics.
"""

import numpy as np
import torch
from torch.nn import functional as F

MULTICLASS_METRICS = [
    "entropy",
    "norm-entropy",
    "margin",
    "margin-top2",
    "deepgini",
    "least-confidence",
    "cross-entropy",
    "second-rank",
]
BINARY_METRICS = ["confusion-distance", "margin", "deepgini", "least-confidence", "binary-entropy", "bce"]


def compute_confusion_distance(probs):
    """
    Calculate the confusion distance of the classifier output. Only for binary classification.

    Original confusion distance:
    (CD) = |0.5 - probs|

    Updated (so that, the higher the value, the closer to the decision boundary the classifier is):
    (CD) = (probs - 0.5) if probs > 0.5 else probs
    """
    # return (0.5 - probs).abs()
    return torch.where(probs > 0.5, probs - 0.5, probs)


def compute_binary_entropy(probs):
    """Calculate the binary entropy of the classifier output."""
    return -probs * np.log2(probs) - (1 - probs) * np.log2(1 - probs)


def compute_entropy(probs):
    """Calculate the entropy of the classifier output."""
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)


def compute_norm_entropy(probs):
    """Calculate the normalized entropy of the classifier output."""
    entropy = compute_entropy(probs)
    return entropy / torch.log(torch.tensor(probs.size(1)).float())


def compute_margin(probs):
    """Calculate the margin of the classifier output. Per each data point, find the difference between the highest and the lowest probability.

    The higher the value, the closer to the decision boundary the classifier is.
    """
    return 1 - (probs.max(dim=1).values - probs.min(dim=1).values)


def compute_margin_top2(probs):
    """Calculate the margin of the classifier output. Per each data point, find the difference between the highest and the second highest probability.

    The higher the value, the closer to the decision boundary the classifier is.
    """
    top_probs, _ = torch.topk(probs, 2)
    return 1 - (top_probs[:, 0] - top_probs[:, 1])


def compute_deepgini(probs):
    """Calculate the deep gini of the classifier output. Metric based on the Gini impurity measure."""
    return 1 - torch.sum(probs**2, dim=1)


def compute_least_confidence(probs):
    """Calculate the least confidence of the classifier output. The lowest probability is the least confident."""
    return 1 - probs.max(dim=1).values


def compute_second_rank(probs):
    """Calculate the second rank of the classifier output. The second highest probability."""
    return torch.topk(probs, 2).values[:, 1]


def compute_cross_entropy_loss(probs, logits):
    """Calculate the cross-entropy loss of the classifier output."""
    return F.cross_entropy(logits, probs, reduction="none")


def compute_bce_loss(probs, logits):
    """Calculate the cross-entropy loss of the classifier output."""
    # TODO check if it is working
    return F.binary_cross_entropy(logits, probs, reduction="none")


def compute_gaussian_loss(probs, logits):
    """Calculate the cross-entropy loss of the classifier output."""
    # TODO check if it is working
    target = torch.full_like(input=probs, fill_value=0.5)  # target is 0.5 (binary)
    var = torch.full_like(input=probs, fill_value=0.01)  # variance is 0.1 (we can test different values)
    return F.gaussian_nll_loss(logits, target, var, reduction="none")


def compute_metric(metric, probs, logits=None):
    """Calculate the specified metric of the classifier output."""
    # avoid problems with log(0) or log(1)
    probs = torch.clamp(probs, 1e-10, 1 - 1e-10)

    metric_functions = {
        "entropy": compute_entropy,
        "confusion-distance": compute_confusion_distance,
        "binary-entropy": compute_binary_entropy,
        "norm-entropy": compute_norm_entropy,
        "margin": compute_margin,
        "margin-top2": compute_margin_top2,
        "deepgini": compute_deepgini,
        "least-confidence": compute_least_confidence,
        "second-rank": compute_second_rank,
    }

    loss_functions = {
        "cross-entropy": compute_cross_entropy_loss,
        "bce": compute_bce_loss,
    }

    if metric in metric_functions:
        return metric_functions[metric](probs)

    if metric in loss_functions:
        if logits is not None:
            return loss_functions[metric](probs, logits)
        return torch.tensor(float("nan"))

    raise ValueError(f"Metric {metric} not supported")
