"""
Metrics for evaluating the classifier output. Can be used for guidance and for visualization purposes.

We receive the classifier output as a tensor of probabilities. The metrics are calculated per each data point in the batch.
If the goal is to minimize the metric, we should return a negative value
If the goal is to maximize the metric, we should return a positive value
"""

import numpy as np
import torch
from torch.nn import functional as F

MULTICLASS_METRICS = [
    "entropy",
    "cross-entropy",
    "margin-top2",
    "deepgini",
    "second-rank",
    "evidential-ambiguity",
    "kl-div-target",
    # "margin",
    # "least-confidence",
]
BINARY_METRICS = ["confusion-distance", "binary-entropy", "margin", "deepgini", "kl-div-target"]  # "least-confidence"
UNCERTAINTY_METRICS = ["mc-dropout-mean"]


def compute_confusion_distance(probs):
    """
    Calculate the confusion distance of the classifier output.

    Original confusion distance (from GASTeN paper)
    (CD) = |0.5 - probs|

    Only for binary classification.
    Goal: minimize
    """
    return -torch.abs(0.5 - probs)


def compute_binary_entropy(probs):
    """Calculate the binary entropy of the classifier output.

    Only for binary classification.
    Goal: maximize
    """
    return -probs * np.log2(probs) - (1 - probs) * np.log2(1 - probs)


def compute_shannon_entropy(probs):
    """Calculate the entropy of the classifier output. It is the Shannon entropy.

    Works for multicass classification.
    It is maximum if all classes have the same probability, meaning that it is not a good metric to find the confusion between specific classes.
    Goal: maximize
    """
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)


def compute_margin(probs):
    """Calculate the margin of the classifier output. Per each data point, find the difference between the highest and the lowest probability.

    Works for multicass and binary classification.
    If zero, all values have the same probabilities, meaning that it is not a good metric to find the confusion between specific classes.
    Goal: minimize
    """
    return -(probs.max(dim=1).values - probs.min(dim=1).values)


def compute_margin_top2(probs):
    """Calculate the margin of the classifier output. Per each data point, find the difference between the highest and the second highest probability.

    Works for multicass and binary classification. In the case of binary classification is the same as the margin metric.
    If zero, we are at the decision boundary between the top 2 classes.
    Goal: minimize
    """
    top_probs, _ = torch.topk(probs, 2)
    return 1 - (top_probs[:, 0] - top_probs[:, 1])


def compute_deepgini(probs):
    """Calculate the deep gini of the classifier output. Metric based on the Gini impurity measure.

    The maximum value means the distribution is completely uniform (maximum impurity) - similar to entropy.
    Works for multiclass and binary classification.
    Goal: maximize
    """
    return 1 - torch.sum(probs**2, dim=1)


def compute_least_confidence(probs):
    """Calculate the least confidence of the classifier output. The lowest probability is the least confident.

    Works for multiclass and binary classification.
    Goal: maximize
    """
    return probs.max(dim=1).values


def compute_second_rank(probs):
    """Calculate the second rank of the classifier output, that is, the second highest probability.

    Goal: maximize
    """
    return torch.topk(probs, 2).values[:, 1]


def compute_evidential_ambiguity(probs):
    """Calculate the evidential ambiguity of the classifier output based on the theory of evidence from Dempster-Shafter.

    Goal: minimize
    """
    K = probs.shape[1]  # n classes
    m = probs - (1 / K)  # confidence
    m[m < 0] = 0
    return -m.sum(dim=1)


def compute_cross_entropy_loss(probs, logits):
    """Calculate the cross-entropy loss of the classifier output.

    Goal: maximize
    """
    return F.cross_entropy(logits, probs, reduction="none")


def compute_probs_kl_divergence(probs, labels_idx):
    """Compute the KL diveregence between the target and the probs. The target is having equal probabilities for the target classes and zero to the remaining ones.

    Goal: minimize
    """
    n = len(labels_idx)
    max_prob = 1 / n
    target = torch.zeros(*probs.shape, device=probs.device)
    for idx in labels_idx:
        target[:, idx] = max_prob
    target = torch.clip(target, min=1e-10)

    return -F.kl_div(probs.log(), target, reduction="batchmean")


def compute_gaussian_loss(probs, logits):
    """Calculate the cross-entropy loss of the classifier output.

    Goal: maximize
    """
    # TODO check if it is working
    target = torch.full_like(input=probs, fill_value=0.5)  # target is 0.5 (binary)
    var = torch.full_like(input=probs, fill_value=0.01)  # variance is 0.1 (we can test different values)
    return F.gaussian_nll_loss(logits, target, var, reduction="none")


def compute_mc_dropout_mean(probs_dropout):
    """Calculate the mean of the variance of the MC dropout probabilities.

    The higher the variance, the higher epistemic uncertainty.
    Goal: maximize
    """
    return probs_dropout.var(dim=0).mean(dim=-1)


def compute_metric(metric, probs, probs_dropout=None, logits=None, labels_idx=None):
    """Calculate the specified metric of the classifier output."""
    # avoid problems with log(0) or log(1)
    probs = torch.clip(probs, 1e-10, 1 - 1e-10)

    # metrics that only need the probabilities
    probs_functions = {
        "entropy": compute_shannon_entropy,
        "confusion-distance": compute_confusion_distance,
        "binary-entropy": compute_binary_entropy,
        "margin": compute_margin,
        "margin-top2": compute_margin_top2,
        "deepgini": compute_deepgini,
        "least-confidence": compute_least_confidence,
        "second-rank": compute_second_rank,
        "evidential-ambiguity": compute_evidential_ambiguity,
    }
    # metrics that need the logits
    loss_functions = {
        "cross-entropy": compute_cross_entropy_loss,
    }
    # metrics that need the target classes
    target_functions = {
        "kl-div-target": compute_probs_kl_divergence,
    }
    # metrics that require multiple forward passes
    uncertainty_functions = {
        "mc-dropout-mean": compute_mc_dropout_mean,
    }

    if metric in probs_functions:
        return probs_functions[metric](probs)

    if metric in loss_functions:
        return loss_functions[metric](probs, logits) if logits is not None else torch.tensor(float("nan"))

    if metric in target_functions:
        return target_functions[metric](probs, labels_idx) if labels_idx is not None else torch.tensor(float("nan"))

    if metric in uncertainty_functions:
        return uncertainty_functions[metric](probs_dropout) if probs_dropout is not None else torch.tensor(float("nan"))

    raise ValueError(f"Metric {metric} not supported")
