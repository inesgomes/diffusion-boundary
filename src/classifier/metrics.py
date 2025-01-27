"""
Metrics for evaluating the classifier output. Can be used for guidance and for visualization purposes.

We receive the classifier output as a tensor of probabilities. The metrics are calculated per each data point in the batch. 
IMPORTANT: all metrics should be implement so that the higher the value, the closer to the decision boundary the classifier is. In the other words, the less confident the classifier is, the higher the metric value should be.
It would also be helpful to have a function that normalizes the metric values to the range [0, 1] for better comparison between different metrics.
"""

import torch

MULTICLASS_METRICS = ["entropy", "norm-entropy", "margin", "margin-top2", "deepgini"]
BINARY_METRICS = ["confusion-distance", "margin", "deepgini"]


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


def compute_metric(metric, probs):
    """Calculate the specified metric of the classifier output."""
    if metric == "entropy":
        return compute_entropy(probs)
    if metric == "confusion-distance":
        return compute_confusion_distance(probs)
    if metric == "norm-entropy":
        return compute_norm_entropy(probs)
    if metric == "margin":
        return compute_margin(probs)
    if metric == "margin-top2":
        return compute_margin_top2(probs)
    if metric == "deepgini":
        return compute_deepgini(probs)
    raise ValueError(f"Metric {metric} not supported")
