"""Toy-dataset figures: decision regions plus the boundary-proximity metrics.

Each figure is one blob dataset under a one-vs-rest logistic regression: the
first panel shows the classes, the rest colour the same points by a metric.
See README.md for the colour and print-scale rules.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (
    BoundaryNorm,
    ListedColormap,
    LogNorm,
    Normalize,
    PowerNorm,
)
from matplotlib.lines import Line2D
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    DrawingArea,
    HPacker,
    TextArea,
    VPacker,
)
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from style import CATEGORICAL, CATEGORICAL_TINTS, TEXT_WIDTH_IN, paper_rc

REPORT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = REPORT_DIR / "figures"

EPS = 1e-9  # keeps log(0) out of the entropy and KLDB sums

CLASS_COLORS = CATEGORICAL
REGION_COLORS = CATEGORICAL_TINTS  # a region sits under the class it predicts
# 'o' is the marker used in the metric panels, so classes avoid it entirely
CLASS_MARKERS = ["s", "^", "D"]  # square, triangle, diamond

POINT_SIZE = 16  # marker area in pt^2, about the most a 1.7 in panel takes
POINT_EDGE_WIDTH = 0.3
GRID_RESOLUTION = 400  # samples per axis for the decision surface

# x0/x1 are incidental, so they undercut PAPER_RC and give the width back to the
# panels. The colourbars keep the shared sizes.
AXIS_LABEL_SIZE = 5
AXIS_TICK_SIZE = 4.5


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def entropy(probs):
    """Shannon entropy per row, in nats; highest where the classes are equiprobable."""
    return -np.sum(probs * np.log(probs + EPS), axis=1)


def kldb(probs, audited):
    """KL divergence from the uniform distribution over ``audited`` to each row of ``probs``.

    No target mass outside the audited classes C reduces it to
    ``-log|C| - (1/|C|) sum_{i in C} log p_i``: the same function as
    ``src.classifier.metrics.compute_kldb``, in numpy over probabilities.
    """
    audited = list(audited)
    return -np.log(len(audited)) - np.log(probs[:, audited] + EPS).mean(axis=1)


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scenario:
    """One toy figure: a blob dataset, and the class subset the metrics audit.

    ``audited`` is a strict subset of the classes where the figure isolates only
    some boundaries of a multiclass fit. ``confusion_distance`` is per scenario
    because each measures the gap to an equal split differently.
    """

    name: str
    centers: int
    cluster_std: float
    class_labels: Sequence[str]
    audited: Sequence[int]
    confusion_distance: Callable[[np.ndarray], np.ndarray]


SCENARIOS = [
    Scenario(
        name="toy-dataset-2classes-2boundaries",
        centers=2,
        cluster_std=4.0,
        class_labels=("A", "B"),
        audited=(0, 1),
        # binary: distance from the equiprobable point, i.e. |p0 - p1| / 2
        confusion_distance=lambda probs: np.abs(probs[:, 0] - 0.5),
    ),
    Scenario(
        name="toy-dataset-3classes-3boundaries",
        centers=3,
        cluster_std=5.0,
        class_labels=("A", "B", "C"),
        audited=(0, 1, 2),
        confusion_distance=lambda probs: np.abs(probs[:, 0] - probs[:, 1] - probs[:, 2]),
    ),
    Scenario(
        # same three-class fit as above, but auditing only the A|B boundary
        name="toy-dataset-3classes-2boundaries",
        centers=3,
        cluster_std=5.0,
        class_labels=("A", "B", "C"),
        audited=(0, 1),
        confusion_distance=lambda probs: np.abs(probs[:, 0] - probs[:, 1]),
    ),
]


def fit_toy_classifier(centers, cluster_std, n_samples=150, random_state=42):
    """Standardised blobs plus the one-vs-rest logistic regression fitted to them.

    ``OneVsRestClassifier`` replaces the removed ``multi_class="ovr"`` argument.
    """
    X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=random_state, cluster_std=cluster_std)
    X_scaled = StandardScaler().fit_transform(X)
    clf = OneVsRestClassifier(LogisticRegression()).fit(X_scaled, y)
    return X_scaled, y, clf


@dataclass
class MetricPanel:
    """One metric panel: the per-point values, its colourbar label, colormap and norm."""

    values: np.ndarray
    label: str
    cmap: str
    norm: Normalize


def metric_panels(probs, scenario):
    """Build the three metric panels for one scenario, each with the norm its range needs."""
    distance = scenario.confusion_distance(probs)
    divergence = kldb(probs, scenario.audited)
    # LogNorm rejects a non-positive vmin; KLDB hits 0 on an exactly uniform posterior.
    return [
        MetricPanel(
            distance,
            "confusion distance (↓)",
            "cividis",
            Normalize(vmin=0, vmax=round(float(distance.max()), 1)),
        ),
        MetricPanel(entropy(probs), "entropy (↑)", "cividis_r", PowerNorm(gamma=0.9)),
        MetricPanel(
            divergence,
            "KLDB (↓)",
            "cividis",
            LogNorm(vmin=max(float(divergence.min()), EPS), vmax=float(divergence.max())),
        ),
    ]


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def centered_row_legend(
    ax,
    labels,
    colors,
    markers,
    *,
    per_row=2,
    bbox_to_anchor=(0.5, -0.06),
    fontsize=None,
    markersize=4,
    marker_box=10,
    entry_sep=12,
    row_sep=1.5,
):
    """Legend filled row-major, with a short trailing row centred.

    matplotlib's own legend is a grid filled column-major, so three entries at
    ncol=2 give "A C / B" and cannot centre the orphan. Hand-packing fixes both.
    ``fontsize=None`` tracks the ambient ``legend.fontsize``.
    """
    fontsize = mpl.rcParams["legend.fontsize"] if fontsize is None else fontsize

    def entry(label, color, marker):
        area = DrawingArea(marker_box, marker_box, 0, 0)
        area.add_artist(
            Line2D(
                [marker_box / 2],
                [marker_box / 2],
                marker=marker,
                linestyle="None",
                markersize=markersize,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.6,
            )
        )
        text = TextArea(label, textprops={"fontsize": fontsize})
        return HPacker(children=[area, text], align="center", pad=0, sep=4)

    entries = [entry(*item) for item in zip(labels, colors, markers)]
    rows = [
        HPacker(children=entries[i : i + per_row], align="center", pad=0, sep=entry_sep)
        for i in range(0, len(entries), per_row)
    ]

    # align="center" centres a trailing row shorter than the others
    box = VPacker(children=rows, align="center", pad=0, sep=row_sep)

    anchored = AnchoredOffsetbox(
        loc="upper center",
        child=box,
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=ax.transAxes,
        frameon=True,
        pad=0.4,
        borderpad=0.0,
    )
    anchored.patch.set_facecolor("white")
    anchored.patch.set_edgecolor("0.7")
    ax.add_artist(anchored)
    return anchored


def _decision_surface(clf, X_scaled, resolution=GRID_RESOLUTION):
    """Predicted class over a grid spanning the data, padded by one unit."""
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    return xx, yy, clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)


def _draw_regions(ax, surface, n_classes):
    """Tint each predicted region neutrally and carry the boundary on a black line."""
    xx, yy, Z = surface
    ax.contourf(
        xx,
        yy,
        Z,
        levels=np.arange(-0.5, n_classes, 1.0),  # one band per class
        cmap=ListedColormap(REGION_COLORS[:n_classes]),
        alpha=1.0,
    )
    ax.contour(
        xx,
        yy,
        Z,
        levels=np.arange(0.5, n_classes - 0.5, 1.0),
        colors="k",
        linewidths=0.8,
        linestyles="-",
    )


def _draw_class_panel(ax, X_scaled, y, class_labels):
    """Scatter the ground-truth classes by colour and shape, with the legend below."""
    y = np.asarray(y)
    for k in range(len(class_labels)):
        mask = y == k
        ax.scatter(
            X_scaled[mask, 0],
            X_scaled[mask, 1],
            color=CLASS_COLORS[k],
            marker=CLASS_MARKERS[k],
            s=POINT_SIZE,
            edgecolor="k",
            linewidths=POINT_EDGE_WIDTH,
        )

    # legend sits below the panel, in the slot the colourbar vacated
    centered_row_legend(
        ax,
        class_labels,
        CLASS_COLORS[: len(class_labels)],
        CLASS_MARKERS[: len(class_labels)],
    )

    # A hidden colourbar reserves the slot: without it this panel would grow
    # taller than the others, which give up that space to their own colourbars.
    n_classes = len(class_labels)
    mappable = ScalarMappable(
        cmap=ListedColormap(CLASS_COLORS[:n_classes]),
        norm=BoundaryNorm(np.arange(-0.5, n_classes, 1.0), n_classes),
    )
    ax.figure.colorbar(mappable, ax=ax, orientation="horizontal", pad=0.05).ax.set_visible(False)


def _draw_metric_panel(ax, X_scaled, panel):
    """Scatter the points coloured by one metric, with its labelled colourbar below."""
    scatter = ax.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=panel.values,
        cmap=panel.cmap,
        norm=panel.norm,
        s=POINT_SIZE,
        edgecolor="k",
        linewidths=POINT_EDGE_WIDTH,
    )
    cbar = ax.figure.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.05)
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.set_label(panel.label)
    cbar.ax.tick_params(length=2.5, width=0.6, direction="out")


def _style_axes(ax):
    """Move the x axis to the top so the colourbar owns the bottom of the panel."""
    ax.xaxis.set_label_position("top")
    ax.xaxis.set_ticks_position("top")
    ax.set_xlabel("x0", fontsize=AXIS_LABEL_SIZE, labelpad=2)
    ax.set_ylabel("x1", fontsize=AXIS_LABEL_SIZE, labelpad=2)
    ax.tick_params(axis="x", labeltop=True, labelbottom=False)
    ax.tick_params(labelsize=AXIS_TICK_SIZE, length=1.5, pad=1.5)


def plot_metrics(
    X_scaled,
    y,
    clf,
    class_labels,
    panels,
    *,
    fig_width=TEXT_WIDTH_IN,
    fig_height=2.2,  # panels come out square at text width; see AXIS_LABEL_SIZE
):
    """Draw the class panel and one panel per metric, all over the same decision regions.

    Each ``MetricPanel`` in ``panels`` brings its own norm, so the figure never
    infers a scale from a panel's position.
    """
    n_classes = len(class_labels)
    if n_classes > len(CLASS_COLORS):
        # The palette stops where it stops validating. See README.md.
        raise ValueError(f"{n_classes} classes exceeds the {len(CLASS_COLORS)} validated palette slots")
    surface = _decision_surface(clf, X_scaled)

    n_panels = len(panels) + 1
    with mpl.rc_context(paper_rc(fig_width)):
        fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, fig_height), squeeze=False)
        axes = axes[0]

        for ax, panel in zip(axes, [None, *panels]):
            _draw_regions(ax, surface, n_classes)
            if panel is None:
                _draw_class_panel(ax, X_scaled, y, class_labels)
            else:
                _draw_metric_panel(ax, X_scaled, panel)
            _style_axes(ax)

        fig.tight_layout()
    return fig, axes


def save_figure(fig, name, folder_path=FIGURES_DIR):
    """Write the figure as PNG and PDF under ``folder_path`` and return both paths.

    Saving runs under the rc context too: matplotlib resolves "serif" against
    ``font.serif`` at save time, so saving outside it embeds the wrong font.
    """
    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)
    stem = folder_path / name

    png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    with mpl.rc_context(paper_rc(fig.get_figwidth())):
        fig.savefig(png, bbox_inches="tight", dpi=300)
        fig.savefig(pdf, bbox_inches="tight")  # pdf.fonttype 42: TrueType, not Type 3
    return png, pdf


def main():
    """Draw every toy scenario, save it, and report the KLDB range behind each figure."""
    for scenario in SCENARIOS:
        X_scaled, y, clf = fit_toy_classifier(scenario.centers, scenario.cluster_std)
        panels = metric_panels(clf.predict_proba(X_scaled), scenario)

        fig, _ = plot_metrics(X_scaled, y, clf, scenario.class_labels, panels)
        png, pdf = save_figure(fig, scenario.name)
        plt.close(fig)

        divergence = panels[-1].values
        print(f"{scenario.name}: KLDB in [{divergence.min():.4f}, {divergence.max():.4f}]")
        print(png, pdf, sep="\n")


if __name__ == "__main__":
    main()
