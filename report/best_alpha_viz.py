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
from style import AXIS, CATEGORICAL, GRIDLINE, INK, INK_SECONDARY, paper_rc

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

VALIDITY_TOL = 0.01  # see select_best_alpha; tight enough that 0.02 moves a pick
KLDB_TOL = 0.10

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
# Unlisted subsets fall back to listing their class names.
SUBSET_LABELS = {
    ("golden retriever", "Labrador retriever"): "Dogs",
    ("leopard", "jaguar", "cheetah"): "Felines",
    ("ruddy turnstone", "red-backed sandpiper", "redshank", "dowitcher"): "Birds",
}

FIG_SIZE_IN = (7.8, 3.3)  # drawn wider than the page; include it at \textwidth


def _class_names(classes_json):
    """Return the audited classes, by primary common name."""
    return tuple(name.split(",")[0].strip() for name in json.loads(classes_json))


def _subset_label(classes_json):
    """Panel title, e.g. "Dogs (2 classes)"."""
    names = _class_names(classes_json)
    return f"{SUBSET_LABELS.get(names, ' · '.join(names))} ({len(names)} classes)"


def _by_config(frame, column=None):
    """Index a per-configuration frame by (classes, classifier)."""
    return {
        (row[CLASSES_COL], row[CLASSIFIER_COL]): row if column is None else row[column] for _, row in frame.iterrows()
    }


def _load_runs():
    """Load the sweep, one row per run."""
    required = [CLASSES_COL, CLASSIFIER_COL, ALPHA_COL, "kldb_25", "kldb_median", "kldb_75", "coverage"]
    # The validity fraction is NaN for at least one run, so it cannot be required.
    optional = [RUN_ID_COL, VALIDITY_COL, "num-images", "seed"]
    df = pd.read_csv(RUNS_CSV)
    return df[required + optional].dropna(subset=required).sort_values(ALPHA_COL)


def _at_validity_peak(config, validity_tol=VALIDITY_TOL):
    """Alphas within ``validity_tol`` of the peak top-|C| subset fraction.

    Past that peak guidance pushes probability mass out of the audited subset,
    so a lower KLDB there is measuring a different boundary. NaN means
    unmeasured, not failed, so those alphas stay in the running.
    """
    validity = config[VALIDITY_COL]
    return config[validity.isna() | (validity >= (1 - validity_tol) * validity.max())]


def _kldb_undominated(config):
    """Alphas no other alpha beats on all three KLDB quartiles at once.

    Reads the whole distribution, not the median alone: an alpha survives unless
    another is at least as close at the 25th, 50th and 75th percentile, and
    strictly closer at one.
    """
    quartiles = config[["kldb_25", "kldb_median", "kldb_75"]].to_numpy()
    keep = [
        not any((other <= row).all() and (other < row).any() for j, other in enumerate(quartiles) if j != i)
        for i, row in enumerate(quartiles)
    ]
    return config[keep]


# Each strategy narrows the alphas; the shared steps below then take the lowest
# median KLDB (within kldb_tol) and break ties on coverage.
STRATEGIES = {
    "validity": (
        _at_validity_peak,
        "the largest top-$|C|$ subset fraction, then the lowest median KLDB",
    ),
    "kldb-coverage": (
        _kldb_undominated,
        "the alphas undominated across the KLDB quartiles, then the lowest median KLDB",
    ),
}


def select_best_alpha(runs=None, strategy="kldb-coverage", kldb_tol=KLDB_TOL, **kwargs):
    """Pick one alpha per (subset, classifier), by one of the STRATEGIES.

    1. ``strategy`` narrows the alphas: ``"validity"`` keeps those at the
       top-|C| subset-fraction peak, ``"kldb-coverage"`` keeps those undominated
       across the KLDB quartiles (it reads nothing but KLDB and coverage).
    2. Take the lowest median KLDB, counting anything within ``kldb_tol`` as tied.
    3. Break ties on coverage, the manifold-fidelity guard.

    Extra keyword arguments go to the strategy (e.g. ``validity_tol``).
    """
    narrow, _ = STRATEGIES[strategy]
    df = _load_runs() if runs is None else runs
    rows = []
    for (classes, classifier), config in df.groupby([CLASSES_COL, CLASSIFIER_COL], sort=False):
        candidates = narrow(config, **kwargs)
        tied = candidates[candidates["kldb_median"] <= (1 + kldb_tol) * candidates["kldb_median"].min()]
        chosen = tied.loc[tied["coverage"].idxmax()]
        rows.append(
            {
                CLASSES_COL: classes,
                CLASSIFIER_COL: classifier,
                "strategy": strategy,
                "alpha": chosen[ALPHA_COL],
                "run_id": chosen[RUN_ID_COL],
                "kldb_25": chosen["kldb_25"],
                "kldb_median": chosen["kldb_median"],
                "kldb_75": chosen["kldb_75"],
                VALIDITY_COL: chosen[VALIDITY_COL],
                "coverage": chosen["coverage"],
                "candidates": ", ".join(f"{a:g}" for a in sorted(candidates[ALPHA_COL])),
            }
        )
    return pd.DataFrame(rows)


def real_kldb_reference(selection=None):
    """Median KLDB of the *real* ImageNet images, per (subset, classifier).

    Read from ``FILESDIR/logs/<run_id>/results_real.parquet``. The real set does
    not depend on alpha, so the alpha* run only keeps reference and marked point
    on one run. Returns an empty frame if the logs are unreachable.
    """
    selection = select_best_alpha() if selection is None else selection
    # The repo reads FILESDIR via python-dotenv, which this report does not
    # depend on; parse the same .env directly.
    files_dir = os.environ.get("FILESDIR") or dict(
        line.split("=", 1) for line in PROJECT_ENV.read_text().splitlines() if "=" in line
    ).get("FILESDIR", "").strip("\"' ")

    rows = []
    for _, pick in selection.iterrows():
        path = Path(files_dir) / "logs" / str(pick["run_id"]) / "results_real.parquet"
        if not files_dir or not path.is_file():
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


def _draw_series(ax_kldb, ax_cov, run, style, pick=None, real_median=None):
    """One classifier's curves in one panel."""
    line = {
        "markerfacecolor": "white",
        "markeredgecolor": style["color"],
        "markeredgewidth": 0.8,
        "zorder": 3,
        "clip_on": False,
        **style,
    }
    diamond = {
        "marker": "D",
        "linestyle": "none",
        "markersize": 4.5,
        "color": style["color"],
        "markeredgewidth": 1.0,
        "zorder": 5,
        "clip_on": False,
    }

    if real_median is not None:
        # The gap to the curve is how far guidance moved the generated set.
        ax_kldb.axhline(real_median, color=style["color"], linestyle=":", linewidth=1.0, zorder=1)
    ax_kldb.fill_between(
        run[ALPHA_COL], run["kldb_25"], run["kldb_75"], color=style["color"], alpha=0.12, linewidth=0, zorder=2
    )
    ax_kldb.plot(run[ALPHA_COL], run["kldb_median"], **line)
    ax_cov.plot(run[ALPHA_COL], run["coverage"], **line)
    if pick is not None:
        ax_kldb.plot([pick["alpha"]], [pick["kldb_median"]], **diamond)
        ax_cov.plot([pick["alpha"]], [pick["coverage"]], **diamond)


def _legend_handles(classifiers, styles, with_reference):
    handles = [
        Line2D(
            [], [], markerfacecolor="white", markeredgewidth=0.8, label=CLASSIFIER_LABELS.get(clf, clf), **styles[clf]
        )
        for clf in classifiers
    ]
    handles.append(
        Line2D([], [], color=INK, marker="D", linestyle="none", markersize=4.5, label=r"selected $\alpha^{*}$")
    )
    if with_reference:
        handles.append(Line2D([], [], color=INK, linestyle=":", linewidth=1.0, label="real ImageNet median KLDB"))
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

    # Panels ordered by subset size, coarse -> fine.
    groups = sorted(df[CLASSES_COL].unique(), key=lambda c: (len(_class_names(c)), c))
    classifiers = sorted(df[CLASSIFIER_COL].unique())
    styles = {clf: SERIES_STYLES[i] for i, clf in enumerate(classifiers)}

    with mpl.rc_context(paper_rc(FIG_SIZE_IN[0])):
        fig, axes = plt.subplots(
            2, len(groups), figsize=FIG_SIZE_IN, sharex=True, sharey="row", squeeze=False, layout="constrained"
        )
        fig.set_layout_engine("constrained", w_pad=0.02, h_pad=0.02, hspace=0.05, wspace=0.04)

        for col, group in enumerate(groups):
            ax_kldb, ax_cov = axes[0, col], axes[1, col]
            panel = df[df[CLASSES_COL] == group]
            for clf in classifiers:
                run = panel[panel[CLASSIFIER_COL] == clf].sort_values(ALPHA_COL)
                if not run.empty:
                    _draw_series(
                        ax_kldb, ax_cov, run, styles[clf], picks.get((group, clf)), real_kldb.get((group, clf))
                    )

            title = f"({string.ascii_lowercase[col]}) {_subset_label(group)}"
            ax_kldb.set_title(textwrap.fill(title, width=32), color=INK, pad=4)
            ax_kldb.set_yscale("log")
            ax_cov.set_ylim(bottom=0)
            ax_cov.set_xticks(sorted(df[ALPHA_COL].unique()))
            ax_cov.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            _style_axes(ax_kldb)
            _style_axes(ax_cov)

        # Rows share a scale, so only the left-hand panel is labelled.
        axes[0, 0].set_ylabel("median KLDB", color=INK)
        axes[1, 0].set_ylabel("coverage", color=INK)
        axes[0, 0].yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        axes[0, 0].yaxis.set_minor_formatter(NullFormatter())
        axes[0, 0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axes[1, 0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        fig.supxlabel(r"guidance strength $\alpha$", fontsize=8.5, color=INK)

        # Anchored below the canvas, not "outside lower center": constrained
        # layout gives that and the supxlabel the same band, and they collide.
        handles = _legend_handles(classifiers, styles, with_reference=bool(real_kldb))
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

    caption = _caption(df, groups, reference, selection["strategy"].iloc[0])
    return stem.with_suffix(".png"), stem.with_suffix(".pdf"), caption


def _caption(df, groups, reference, strategy):
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
        "KLDB. The KLDB axis is shared across subsets. Each point is one run over "
        f"N = {n_images} generated images (seed {seeds}). Diamonds mark the selected "
        f"$\\alpha^{{*}}$ per configuration: {STRATEGIES[strategy][1]} within "
        f"{KLDB_TOL:.0%}, ties broken by coverage."
    )
    if reference.empty:
        return caption
    return caption + (
        " Dotted lines give the median KLDB of the real ImageNet images of the audited "
        f"classes under the same classifier (N = {reference['n_real'].max()}); the gap "
        "to each curve is how far guidance moved the generated set toward the boundary."
    )


def main():
    """Draw the figure with the default strategy and print every strategy's picks."""
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
    for name in STRATEGIES:
        table = select_best_alpha(strategy=name) if name != selection["strategy"][0] else selection
        table = table.assign(
            subset=table[CLASSES_COL].map(_subset_label),
            classifier=table[CLASSIFIER_COL].map(lambda c: CLASSIFIER_LABELS.get(c, c)),
        ).sort_values(["subset", "classifier"])
        print(f"\nstrategy: {name}")
        print(table[columns].to_string(index=False, float_format="{:.3f}".format))
    print(f"\n{caption}")


if __name__ == "__main__":
    main()
