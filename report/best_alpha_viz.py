"""KLDB and coverage over the guidance sweep, plus the alpha* selection strategies.

See README.md for the colour and print-scale rules.
"""

import json
import os
import string
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from style import (
    AXIS,
    CATEGORICAL,
    GRIDLINE,
    INK,
    INK_SECONDARY,
    TEXT_WIDTH_IN,
    paper_rc,
)

REPORT_DIR = Path(__file__).resolve().parent
RUNS_CSV = REPORT_DIR / "wandb" / "wandb-runs.csv"
FIGURES_DIR = REPORT_DIR / "figures"
PROJECT_ENV = REPORT_DIR.parent / ".env"

CLASSES_COL = "diffusion.classes"
CLASSIFIER_COL = "classsifier"  # misspelled in the exported W&B table
ALPHA_COL = "diffusion.alpha"
RUN_ID_COL = "ID"
# Fraction of images whose top-|C| classes all lie inside the audited subset.
VALIDITY_COL = "topk-subset_fraction"

KLDB_TOL = 0.10  # median KLDB within this of the best counts as tied

# The first two palette slots. Marker and linestyle repeat the distinction, so
# identity never rests on hue alone.
SERIES_STYLES = [
    {"color": CATEGORICAL[0], "marker": "o", "linestyle": "-"},
    {"color": CATEGORICAL[1], "marker": "s", "linestyle": "--"},
]
CLASSIFIER_LABELS = {
    "google/vit-base-patch16-224": "ViT-B/16",
    "microsoft/resnet-50": "ResNet-50",
}
COVERAGE_SOURCE = "microsoft/resnet-50"  # the one run of a pair whose coverage is trusted
# Unlisted subsets fall back to listing their class names.
SUBSET_LABELS = {
    ("golden retriever", "Labrador retriever"): "Retrievers",
    ("timber wolf", "Eskimo dog"): "Canines",
    ("leopard", "jaguar", "cheetah"): "Felines",
    ("ruddy turnstone", "red-backed sandpiper", "redshank", "dowitcher"): "Birds",
}

# One column per subset. Both dimensions scale with the column count: drawing
# wider is what paper_rc trades for smaller type on the page, so the drawn height
# has to grow in step to keep the printed figure ~2.9 in tall.
PANEL_WIDTH_IN = 2.6
PANEL_HEIGHT_IN = 1.1

# The unguided run is a baseline, not the first point of the sweep: it sits in
# its own slot left of the gap, detached from the curve.
BASELINE_ALPHA = 0.0
BASELINE_X = -0.02
BASELINE_DODGE = 0.01  # classifiers sit side by side in the slot, not on top of each other
BASELINE_LABEL = "no\nguidance"


def _class_names(classes_json):
    """Return the audited classes, by primary common name."""
    return tuple(name.split(",")[0].strip() for name in json.loads(classes_json))


def _subset_label(classes_json):
    """Panel title, e.g. "Retrievers (|C|=2)"."""
    names = _class_names(classes_json)
    return f"{SUBSET_LABELS.get(names, ' · '.join(names))} ($|C|={len(names)}$)"


def _by_config(frame, column=None):
    """Index a per-configuration frame by (classes, classifier)."""
    return {
        (row[CLASSES_COL], row[CLASSIFIER_COL]): row if column is None else row[column] for _, row in frame.iterrows()
    }


def _load_runs():
    """Load the sweep, one row per run, with the unguided coverage deduplicated."""
    required = [CLASSES_COL, CLASSIFIER_COL, ALPHA_COL, "kldb_25", "kldb_median", "kldb_75", "coverage"]
    # The validity fraction is NaN for at least one run, so it cannot be required.
    optional = [RUN_ID_COL, VALIDITY_COL, "num-images", "seed"]
    df = pd.read_csv(RUNS_CSV)[required + optional].dropna(subset=required)

    # Unguided generation does not depend on the classifier, so its coverage is
    # one number per subset and the logged ViT copy is unreliable: take
    # COVERAGE_SOURCE's. Guided coverage genuinely differs and is left alone. A
    # subset with no unguided source run keeps its own value.
    unguided = df[ALPHA_COL] == BASELINE_ALPHA
    source = df[unguided & (df[CLASSIFIER_COL] == COVERAGE_SOURCE)].set_index(CLASSES_COL)["coverage"]
    shared = source.reindex(df.loc[unguided, CLASSES_COL]).to_numpy()
    df.loc[unguided, "coverage"] = pd.Series(shared, index=df.index[unguided]).fillna(df.loc[unguided, "coverage"])
    return df.sort_values(ALPHA_COL)


SELECTION_RULE = "the lowest median KLDB, then the widest coverage among the alphas within"


def select_best_alpha(runs=None, kldb_tol=KLDB_TOL):
    """Pick one alpha per (subset, classifier) from KLDB and coverage alone.

    Among the alphas whose median KLDB is within ``kldb_tol`` of the best any
    alpha reaches for that configuration, take the one with the widest coverage.
    One rule over both objectives: the tolerance states how much boundary
    proximity a gain in manifold fidelity is worth.
    """
    df = _load_runs() if runs is None else runs
    rows = []
    for (classes, classifier), config in df.groupby([CLASSES_COL, CLASSIFIER_COL], sort=False):
        tied = config[config["kldb_median"] <= (1 + kldb_tol) * config["kldb_median"].min()]
        chosen = tied.loc[tied["coverage"].idxmax()]
        rows.append(
            {
                CLASSES_COL: classes,
                CLASSIFIER_COL: classifier,
                "alpha": chosen[ALPHA_COL],
                "run_id": chosen[RUN_ID_COL],
                "kldb_25": chosen["kldb_25"],
                "kldb_median": chosen["kldb_median"],
                "kldb_75": chosen["kldb_75"],
                VALIDITY_COL: chosen[VALIDITY_COL],
                "coverage": chosen["coverage"],
                "candidates": ", ".join(f"{a:g}" for a in sorted(tied[ALPHA_COL])),
            }
        )
    return pd.DataFrame(rows)


def files_dir():
    """Return FILESDIR, the run-log root holding ``logs/<run_id>/results_*.parquet``.

    The repo reads it via python-dotenv, which this report does not depend on;
    parse the same .env directly. Empty string if it is set nowhere.
    """
    return os.environ.get("FILESDIR") or dict(
        line.split("=", 1) for line in PROJECT_ENV.read_text().splitlines() if "=" in line
    ).get("FILESDIR", "").strip("\"' ")


def real_kldb_reference(selection=None):
    """Median KLDB of the *real* ImageNet images, per (subset, classifier).

    Read from ``FILESDIR/logs/<run_id>/results_real.parquet``. The real set does
    not depend on alpha, so the alpha* run only keeps reference and marked point
    on one run. Returns an empty frame if the logs are unreachable.
    """
    selection = select_best_alpha() if selection is None else selection
    files_dir_path = files_dir()

    rows = []
    for _, pick in selection.iterrows():
        path = Path(files_dir_path) / "logs" / str(pick["run_id"]) / "results_real.parquet"
        if not files_dir_path or not path.is_file():
            print(f"no real KLDB reference: {path} unreadable", file=sys.stderr)
            continue
        kldb = pd.read_parquet(path, columns=["kldb"])["kldb"]
        rows.append(
            {
                CLASSES_COL: pick[CLASSES_COL],
                CLASSIFIER_COL: pick[CLASSIFIER_COL],
                "real_kldb_median": float(kldb.median()),
                "n_real": len(kldb),
            }
        )
    return pd.DataFrame(rows)


def _style_axes(ax):
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=AXIS, length=2.5, pad=2, labelcolor=INK_SECONDARY)
    for side, visible in (("top", False), ("right", False), ("left", True), ("bottom", True)):
        ax.spines[side].set_visible(visible)
        ax.spines[side].set_color(AXIS)


def _draw_series(ax_kldb, ax_cov, run, style, pick=None, real_median=None, baseline_x=BASELINE_X):
    """One classifier's curves in one panel."""
    # Sizes in points are drawn-size, so they need the same scale paper_rc applies
    # to the type; without it a wider figure prints thinner marks.
    scale = ax_kldb.figure.get_figwidth() / TEXT_WIDTH_IN
    line = {
        "markerfacecolor": "white",
        "markeredgecolor": style["color"],
        "markeredgewidth": 0.8 * scale,
        "zorder": 3,
        "clip_on": False,
        **style,
    }
    diamond = {
        "marker": "D",
        "linestyle": "none",
        "markersize": 4.5 * scale,
        "color": style["color"],
        "markeredgewidth": 1.0 * scale,
        "zorder": 5,
        "clip_on": False,
    }

    if real_median is not None:
        # The gap to the curve is how far guidance moved the generated set.
        ax_kldb.axhline(real_median, color=style["color"], linestyle=":", linewidth=1.0 * scale, zorder=1)

    guided = run[run[ALPHA_COL] > BASELINE_ALPHA]
    ax_kldb.fill_between(
        guided[ALPHA_COL], guided["kldb_25"], guided["kldb_75"], color=style["color"], alpha=0.12, linewidth=0, zorder=2
    )
    ax_kldb.plot(guided[ALPHA_COL], guided["kldb_median"], **line)
    ax_cov.plot(guided[ALPHA_COL], guided["coverage"], **line)

    baseline = run[run[ALPHA_COL] == BASELINE_ALPHA]
    if not baseline.empty:
        # No curve to band, so the IQR becomes whiskers on the lone point.
        row = baseline.iloc[0]
        point = {**line, "linestyle": "none"}
        ax_kldb.errorbar(
            baseline_x,
            row["kldb_median"],
            yerr=[[row["kldb_median"] - row["kldb_25"]], [row["kldb_75"] - row["kldb_median"]]],
            capsize=2 * scale,
            elinewidth=1.0 * scale,
            capthick=0.8 * scale,
            **point,
        )
        # Undodged, and unfilled so the markers stack rather than hide each other:
        # unguided coverage is one number, the same for every classifier.
        ax_cov.plot(BASELINE_X, row["coverage"], **{**point, "markerfacecolor": "none"})

    if pick is not None:
        alpha_is_baseline = pick["alpha"] == BASELINE_ALPHA
        ax_kldb.plot([baseline_x if alpha_is_baseline else pick["alpha"]], [pick["kldb_median"]], **diamond)
        ax_cov.plot([BASELINE_X if alpha_is_baseline else pick["alpha"]], [pick["coverage"]], **diamond)


def _legend_handles(classifiers, styles, with_reference, scale=1.0):
    handles = [
        Line2D(
            [],
            [],
            markerfacecolor="white",
            markeredgewidth=0.8 * scale,
            label=CLASSIFIER_LABELS.get(clf, clf),
            **styles[clf],
        )
        for clf in classifiers
    ]
    handles.append(
        Line2D([], [], color=INK, marker="D", linestyle="none", markersize=4.5 * scale, label=r"selected $\alpha^{*}$")
    )
    if with_reference:
        handles.append(
            Line2D([], [], color=INK, linestyle=":", linewidth=1.0 * scale, label="real ImageNet median KLDB")
        )
    return handles


def plot_kldb_coverage_over_alpha(selection=None, show_real_reference=False):
    """Draw the alpha sweep, save it as PNG + PDF, and return both paths and a caption.

    ``show_real_reference`` adds a dotted line per classifier at the real
    ImageNet median KLDB; it needs FILESDIR, so it is off by default.
    """
    df = _load_runs()
    selection = select_best_alpha(df) if selection is None else selection
    picks = _by_config(selection)
    reference = real_kldb_reference(selection) if show_real_reference else pd.DataFrame()
    real_kldb = _by_config(reference, "real_kldb_median")

    # Panels ordered by subset size, coarse -> fine, then by the name on the panel.
    groups = sorted(df[CLASSES_COL].unique(), key=lambda c: (len(_class_names(c)), _subset_label(c)))
    classifiers = sorted(df[CLASSIFIER_COL].unique())
    styles = {clf: SERIES_STYLES[i] for i, clf in enumerate(classifiers)}

    # The baseline gets its own tick, off the sweep and past a gap.
    guided_alphas = sorted(a for a in df[ALPHA_COL].unique() if a > BASELINE_ALPHA)
    has_baseline = (df[ALPHA_COL] == BASELINE_ALPHA).any()
    ticks = ([BASELINE_X] if has_baseline else []) + guided_alphas
    tick_labels = ([BASELINE_LABEL] if has_baseline else []) + [f"{a:g}" for a in guided_alphas]
    # Side by side within the slot, centred on the tick.
    baseline_x = {
        clf: BASELINE_X + (i - (len(classifiers) - 1) / 2) * BASELINE_DODGE for i, clf in enumerate(classifiers)
    }
    x_limits = (min(baseline_x.values()) - 0.008, guided_alphas[-1] + 0.008)

    fig_size = (PANEL_WIDTH_IN * len(groups), PANEL_HEIGHT_IN * len(groups))
    with mpl.rc_context(paper_rc(fig_size[0])):
        fig, axes = plt.subplots(
            2, len(groups), figsize=fig_size, sharex=True, sharey="row", squeeze=False, layout="constrained"
        )
        fig.set_layout_engine("constrained", w_pad=0.02, h_pad=0.02, hspace=0.05, wspace=0.04)

        for col, group in enumerate(groups):
            ax_kldb, ax_cov = axes[0, col], axes[1, col]
            panel = df[df[CLASSES_COL] == group]
            for clf in classifiers:
                run = panel[panel[CLASSIFIER_COL] == clf].sort_values(ALPHA_COL)
                if not run.empty:
                    _draw_series(
                        ax_kldb,
                        ax_cov,
                        run,
                        styles[clf],
                        picks.get((group, clf)),
                        real_kldb.get((group, clf)),
                        baseline_x[clf],
                    )

            title = f"({string.ascii_lowercase[col]}) {_subset_label(group)}"
            ax_kldb.set_title(textwrap.fill(title, width=32), color=INK, pad=4)
            ax_kldb.set_yscale("log")
            ax_cov.set_xticks(ticks, labels=tick_labels)
            ax_cov.set_xlim(x_limits)
            _style_axes(ax_kldb)
            _style_axes(ax_cov)

        # Rows share a scale, so only the left-hand panel is labelled. Both rows'
        # limits come from the data rather than autoscale: on the log row it
        # overshoots by a factor once the baseline error bars are in the axes, and
        # any set_ylim call freezes a shared row at the first column it saw.
        axes[0, 0].set_ylim(df["kldb_25"].min() / 1.25, df["kldb_75"].max() * 1.25)
        axes[1, 0].set_ylim(0, df["coverage"].max() * 1.08)
        axes[0, 0].set_ylabel("median KLDB", color=INK)
        axes[1, 0].set_ylabel("coverage", color=INK)
        axes[0, 0].yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        axes[0, 0].yaxis.set_minor_formatter(NullFormatter())
        axes[0, 0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axes[1, 0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        # rcParams, not a literal: supxlabel has to scale with the drawn width
        # like every other label.
        fig.supxlabel(r"guidance strength $\alpha$", fontsize=mpl.rcParams["axes.labelsize"], color=INK)

        # Anchored below the canvas, not "outside lower center": constrained
        # layout gives that and the supxlabel the same band, and they collide.
        scale = fig_size[0] / TEXT_WIDTH_IN
        handles = _legend_handles(classifiers, styles, bool(real_kldb), scale)
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.01),
            bbox_transform=fig.transFigure,
            ncol=len(handles),
            frameon=False,
            handlelength=2.4,
            columnspacing=2.0,
            borderpad=0.0,
            labelcolor=INK,
        )

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        # UTC then back to local: aware, but still reads as local time.
        stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
        stem = FIGURES_DIR / f"kldb_coverage_over_alphas_{stamp}"
        fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.01)
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)

    caption = _caption(df, groups, reference)
    return stem.with_suffix(".png"), stem.with_suffix(".pdf"), caption


def _caption(df, groups, reference):
    """Build the figure caption, with the per-run N spelled out."""
    subsets = "; ".join(
        f"({string.ascii_lowercase[i]}) {_subset_label(g)} - {' · '.join(_class_names(g))}"
        for i, g in enumerate(groups)
    )
    n_images = "/".join(str(int(n)) for n in sorted(df["num-images"].unique()))
    seeds = ", ".join(str(int(s)) for s in sorted(df["seed"].unique()))
    caption = (
        "Median KLDB (top, log scale) and coverage (bottom) as a function of the "
        f"guidance strength $\\alpha$, for each class subset: {subsets}. Shaded "
        "bands give the interquartile range (25th-75th percentile) of the per-image "
        "KLDB; for the unguided run, detached at the left, the IQR is drawn as "
        "whiskers. The KLDB axis is shared across subsets. Each point is one run over "
        f"N = {n_images} generated images (seed {seeds}). Diamonds mark the selected "
        f"$\\alpha^{{*}}$ per configuration: {SELECTION_RULE} {KLDB_TOL:.0%} of it."
    )
    if reference.empty:
        return caption
    return caption + (
        " Dotted lines give the median KLDB of the real ImageNet images of the audited "
        f"classes under the same classifier (N = {reference['n_real'].max()}); the gap "
        "to each curve is how far guidance moved the generated set toward the boundary."
    )


def main():
    """Draw the figure and print the selected alpha* per configuration."""
    selection = select_best_alpha()
    png, pdf, caption = plot_kldb_coverage_over_alpha(selection)
    print(png, pdf, sep="\n")

    columns = [
        "subset",
        "classifier",
        "alpha",
        "kldb_median",
        "kldb_25",
        "kldb_75",
        VALIDITY_COL,
        "coverage",
        "candidates",
    ]
    table = selection.assign(
        subset=selection[CLASSES_COL].map(_subset_label),
        classifier=selection[CLASSIFIER_COL].map(lambda c: CLASSIFIER_LABELS.get(c, c)),
    ).sort_values(["subset", "classifier"])
    print(f"\n{table[columns].to_string(index=False, float_format='{:.3f}'.format)}")
    print(f"\n{caption}")


if __name__ == "__main__":
    main()
