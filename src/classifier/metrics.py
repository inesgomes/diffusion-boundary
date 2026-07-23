"""
Metrics for evaluating the classifier output. Can be used for guidance and for visualization purposes.

We receive the classifier output as a tensor of probabilities. The metrics are calculated per each data point in the batch.
If the goal is to minimize the metric, we should return a negative value
If the goal is to maximize the metric, we should return a positive value
"""

import math

import numpy as np
import torch
from torch.nn import functional as F

# Metrics computed per sample, stored in the results dataframe and aggregated for the run.
# The guidance metric must appear here: the visualizations sort the samples by that column.
# Everything compute_metric knows how to calculate is still available for guidance; these lists
# only decide what gets computed for every sample and reported.
MULTICLASS_METRICS = [
    "entropy",
    "kldb",
    # the subset margin, not the global one: it measures the boundary actually being audited,
    # and unlike margin-top2 it is read off the raw logits, so it survives calibration
    "logit-margin-subset",
    "topk-subset",
    # dropped from reporting: margin-top2, logit-margin, deepgini, second-rank,
    # evidential-ambiguity, kldb_scaled, gaussian-target, margin, least-confidence
]
BINARY_METRICS = [
    "entropy",
    "kldb",
    # with two audited classes this is the global logit margin as well
    "logit-margin-subset",
    "topk-subset",
    # dropped from reporting: margin-top2, logit-margin, confusion-distance, binary-entropy,
    # deepgini, kldb_scaled, margin, least-confidence
]
UNCERTAINTY_METRICS = ["mc-dropout-mean"]

FRACTION_METRICS = ["topk-subset"]


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
    return -(top_probs[:, 0] - top_probs[:, 1])


def compute_logit_margin(logits, labels_idx=None):  # pylint: disable=unused-argument
    """Top-2 margin of the raw logits: sorted(logits)[-1] - sorted(logits)[-2].

    Distance to the nearest decision boundary in logit space, over all classes. Taken on the
    model's own logits, before any temperature scaling, so the value carries no assumption about
    the calibration fit. Zero means the top two classes are tied.
    Goal: minimize
    """
    top_logits, _ = torch.topk(logits, 2, dim=1)
    return -(top_logits[:, 0] - top_logits[:, 1])


def compute_logit_margin_subset(logits, labels_idx=None):
    """Top-2 margin of the raw logits restricted to the audited classes in ``labels_idx``.

    Same quantity as compute_logit_margin, but the top two are taken within the audited subset C,
    so it measures proximity to the boundary actually being audited rather than to whichever
    boundary happens to be nearest. Undefined for fewer than two audited classes.
    Goal: minimize
    """
    if labels_idx is None or len(labels_idx) < 2:
        return torch.full((logits.shape[0],), float("nan"), device=logits.device)
    top_logits, _ = torch.topk(logits[:, labels_idx], 2, dim=1)
    return -(top_logits[:, 0] - top_logits[:, 1])


def compute_topk_subset(probs, labels_idx, logits=None):  # pylint: disable=unused-argument
    """Return 1 if the top-|C| classes all lie inside the audited subset C = ``labels_idx``, else 0.

    An indicator, so it is aggregated as a fraction (see FRACTION_METRICS), and its gradient is
    zero almost everywhere, so it cannot be used for guidance.
    Goal: maximize
    """
    if not labels_idx or any(i is None for i in labels_idx):
        return torch.full((probs.shape[0],), float("nan"), device=probs.device)
    k = len(labels_idx)
    if k > probs.shape[1]:
        return torch.full((probs.shape[0],), float("nan"), device=probs.device)
    topk_idx = torch.topk(probs, k, dim=1).indices  # (N, |C|)
    in_subset = torch.zeros(probs.shape[1], dtype=torch.bool, device=probs.device)
    in_subset[torch.as_tensor(labels_idx, device=probs.device)] = True
    return in_subset[topk_idx].all(dim=1).float()


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
    return 1 - probs.max(dim=1).values


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


def compute_kldb(labels_idx, logits):
    """Compute KLDB, the KL divergence from the target distribution to the classifier output.

    The target is uniform over the audited classes (``labels_idx``) and zero elsewhere, which
    reduces the divergence to its closed form:

        KLDB = -log|C| - (1/|C|) * sum_{i in C} log p_i
    """
    if logits is None:
        raise ValueError(
            "KLDB requires logits: the closed form needs log_softmax over the full class vector. "
            "Pass logits=... so the guidance and the evaluation path compute the same function."
        )
    k = len(labels_idx)
    log_probs = F.log_softmax(logits, dim=1)
    return -math.log(k) - log_probs[:, labels_idx].sum(dim=1) / k


def compute_kldb_decomposition(logits, labels_idx):
    """Split KLDB into balance + mass, with C = ``labels_idx`` and p = softmax over all classes.

        m       = sum_{i in C} p_i
        balance = -log|C| - mean_{i in C} log(p_i / m)  = KL(uniform_C || p renormalised in C) >= 0
        mass    = -log m                                                                        >= 0

    balance + mass == KLDB exactly. In log space so the tiny off-C probabilities do not underflow.
    Returns ``(m, balance, mass)``, or ``None`` if C is empty or has a missing index.
    """
    if not labels_idx or any(i is None for i in labels_idx):
        return None
    k = len(labels_idx)
    log_p = F.log_softmax(logits, dim=1)  # log p over the full vector
    log_p_c = log_p[:, labels_idx]  # (N, |C|)
    log_m = torch.logsumexp(log_p_c, dim=1)  # log sum_{i in C} p_i
    mass = -log_m
    # balance = -log k - mean(log q_i) = -log k - (mean log p_i - log m)
    balance = -math.log(k) - log_p_c.mean(dim=1) + log_m
    return torch.exp(log_m), balance, mass


def compute_probs_kl_divergence(probs, labels_idx, logits=None):  # pylint: disable=unused-argument
    """Compute the KL diveregence between the target and the probs. The target is having equal probabilities for the target classes and zero to the remaining ones.

    Goal: minimize
    """
    return -compute_kldb(labels_idx, logits)


def compute_probs_kl_divergence_scaled(probs, labels_idx, logits=None):
    """Compute the KL diveregence between the target and the probs. The target is having equal probabilities for the target classes and zero to the remaining ones.

    Goal: minimize
    """
    eps = 1e-10
    k = len(labels_idx)
    C = probs.shape[1]

    gamma = (torch.log(torch.tensor(float(k))) / torch.log(torch.tensor(float(C)))) / 2.0

    kl = compute_kldb(labels_idx, logits)
    kl_scaled = (kl + eps) ** gamma
    return -kl_scaled


def compute_gaussian_loss(probs, logits):
    """Calculate the gaussian loss of the classifier output.

    Goal: maximize
    """
    # TODO check if it is working
    target = torch.full_like(input=probs, fill_value=0.5)  # target is 0.5 (binary)
    var = torch.full_like(input=probs, fill_value=0.01)  # variance is 0.1 (we can test different values)
    return F.gaussian_nll_loss(logits, target, var, reduction="none")


def compute_ideal_gaussian_loss(probs, labels_idx, logits=None):  # pylint: disable=unused-argument
    """Calculate the gaussian loss when comparing to an ideal value (that is our target).

    Goal: minimize
    """
    var = torch.full_like(input=probs, fill_value=0.05)  # consider receiving this value as an argument
    target = torch.zeros(*probs.shape, device=probs.device)
    for idx in labels_idx:
        target[:, idx] = 1 / len(labels_idx)
    target = torch.clip(target, min=1e-10)

    # ideal loss: target vs target
    ideal_loss = F.gaussian_nll_loss(target, target, var=var, reduction="none")
    # current loss: probs vs target
    loss = F.gaussian_nll_loss(probs, target, var=var, reduction="none")
    # my loss
    return -(loss - ideal_loss).sum(dim=1)


def compute_mc_dropout_mean(probs_dropout):
    """Calculate the mean of the variance of the MC dropout probabilities.

    The higher the variance, the higher epistemic uncertainty.
    Goal: maximize
    """
    return probs_dropout.var(dim=0).mean(dim=-1)


def compute_metric(metric, probs, probs_dropout=None, logits=None, labels_idx=None, raw_logits=None):
    """Calculate the specified metric of the classifier output.

    ``logits`` are the ones ``predict`` returns, temperature-scaled when the classifier is
    calibrated. ``raw_logits`` are the model's own, from ``BaseClassifier.raw_logits``; the logit
    margins need those so they do not depend on the calibration fit.
    """
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
    # metrics that need the target classes
    target_functions = {
        "kldb": compute_probs_kl_divergence,
        "kldb_scaled": compute_probs_kl_divergence_scaled,
        "gaussian-target": compute_ideal_gaussian_loss,
        "topk-subset": compute_topk_subset,
    }
    # metrics that need the model's own logits, before any temperature scaling
    raw_logit_functions = {
        "logit-margin": compute_logit_margin,
        "logit-margin-subset": compute_logit_margin_subset,
    }
    # metrics that require multiple forward passes
    uncertainty_functions = {
        "mc-dropout-mean": compute_mc_dropout_mean,
    }

    if metric in probs_functions:
        return probs_functions[metric](probs)

    if metric in raw_logit_functions:
        if raw_logits is None:
            return torch.tensor(float("nan"))
        return raw_logit_functions[metric](raw_logits, labels_idx)

    if metric in target_functions:
        if labels_idx is None:
            return torch.tensor(float("nan"))
        # pass logits so KLDB can use log_softmax (stable) instead of softmax().log()
        return target_functions[metric](probs, labels_idx, logits=logits)

    if metric in uncertainty_functions:
        return uncertainty_functions[metric](probs_dropout) if probs_dropout is not None else torch.tensor(float("nan"))

    raise ValueError(f"Metric {metric} not supported")
