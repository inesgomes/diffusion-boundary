"""ECDF of the per-image KLDB: the unguided run against the selected alpha*.

The sweep figure reports the median and the IQR per alpha; this one shows the
whole distribution, so a shift that moves the median can be told apart from one
that only moves a tail. See README.md for the colour and print-scale rules.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from best_alpha_viz import (
    ALPHA_COL,
    CLASSES_COL,
    CLASSIFIER_COL,
    CLASSIFIER_LABELS,
    FIGURES_DIR,
    RUN_ID_COL,
    RUNS_CSV,
    SERIES_STYLES,
    VALIDITY_COL,
    _style_axes,
    _subset_label,
    files_dir,
)
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from style import GRIDLINE, INK, INK_SECONDARY, paper_rc

# The pair the report compares: same classifier, same subset, same seed, alpha
# apart. Both are rows of wandb-runs.csv, and alpha* is what select_best_alpha
# picks for that configuration.
BASELINE_RUN_ID = "2df847i6"
BEST_RUN_ID = "u00uyam7"

# The per-image column of results_synthetic.parquet this figure is about: KLDB,
# the metric both runs' guidance was configured to optimise.
METRIC = "kldb"

FIG_SIZE_IN = (6.9, 2.7)  # drawn at text width; include it at \textwidth

# Deciles, marked on each curve: markers, not colour, carry identity in print.
DECILES = np.arange(0.1, 0.91, 0.1)

# The real images are a reference, not a series, so they wear ink and a dotted
# line rather than a palette slot - as in the sweep figure.
REAL_STYLE = {"color": INK_SECONDARY, "linestyle": ":", "linewidth": 1.0}


def _run_row(runs, run_id):
    """Return the wandb-runs.csv row for ``run_id``."""
    rows = runs[runs[RUN_ID_COL] == run_id]
    if rows.empty:
        raise SystemExit(f"run {run_id} is not in {RUNS_CSV}")
    return rows.iloc[0]


def _kldb_values(run_id, kind):
    """Per-image KLDB of one run, from ``FILESDIR/logs/<run_id>/results_<kind>.parquet``."""
    path = Path(files_dir()) / "logs" / str(run_id) / f"results_{kind}.parquet"
    if not path.is_file():
        raise SystemExit(f"{path} unreadable: set FILESDIR to the run-log root")
    return pd.read_parquet(path, columns=[METRIC])[METRIC].to_numpy(dtype=float)


def _ecdf(values):
    """Return the step function (x, F) with F(x) the fraction of ``values`` at most x."""
    x = np.sort(values)
    return x, np.arange(1, len(x) + 1) / len(x)


def _ecdf_gap(a, b):
    """Return F_b - F_a evaluated on the pooled sample: the signed vertical gap between two ECDFs."""
    grid = np.sort(np.concatenate([a, b]))
    return np.searchsorted(np.sort(b), grid, "right") / len(b) - np.searchsorted(np.sort(a), grid, "right") / len(a)


def _ks_statistic(a, b):
    """Two-sample Kolmogorov-Smirnov statistic: the largest vertical gap between two ECDFs."""
    return float(np.max(np.abs(_ecdf_gap(a, b))))


def _dominates(a, b):
    """Return True if ``b`` is stochastically smaller than ``a`` everywhere, i.e. the ECDFs never cross."""
    return bool(np.all(_ecdf_gap(a, b) >= 0))


def _draw_ecdf(ax, values, style, label, with_deciles=True):
    """One ECDF, drawn as a step curve with its deciles marked."""
    x, y = _ecdf(values)
    # The curve carries no marker of its own: with one step per image it would
    # be 2500 of them. Identity rests on the linestyle and on the deciles below.
    line = {key: value for key, value in style.items() if key != "marker"}
    # where="post": F is right-continuous, so it jumps *at* each observation.
    ax.step(x, y, where="post", label=label, zorder=3, **line)
    if with_deciles:
        ax.plot(
            np.quantile(values, DECILES),
            DECILES,
            marker=style["marker"],
            linestyle="none",
            markersize=3.0,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=0.8,
            zorder=4,
        )


def _annotate_median_shift(ax, baseline_median, best_median):
    """Draw the median guide and the arrow from the unguided median to the alpha* one."""
    ax.axhline(0.5, color=INK_SECONDARY, linestyle=(0, (1, 3)), linewidth=0.6, zorder=1)
    ax.annotate(
        "",
        xy=(best_median, 0.5),
        xytext=(baseline_median, 0.5),
        arrowprops={"arrowstyle": "-|>", "color": INK, "linewidth": 0.8, "shrinkA": 0, "shrinkB": 0},
        zorder=6,
    )
    change = best_median / baseline_median - 1
    ax.annotate(
        f"median {baseline_median:.2f} $\\rightarrow$ {best_median:.2f} ({change:+.0%})",
        # Left of the arrow tip, on the median line: the one band the label can
        # sit in without crossing a curve, since both are still below 50% there.
        xy=(best_median, 0.5),
        xytext=(-6, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        color=INK,
        fontsize="small",
        zorder=6,
    )


def plot_kldb_ecdf(baseline_run_id=BASELINE_RUN_ID, best_run_id=BEST_RUN_ID, show_real=True):
    """Draw the ECDF pair, save it as PNG + PDF, and return both paths, a caption and the stats.

    ``show_real`` adds the real ImageNet images of the audited classes as a
    dotted reference.
    """
    runs = pd.read_csv(RUNS_CSV)
    baseline, best = _run_row(runs, baseline_run_id), _run_row(runs, best_run_id)

    if baseline[CLASSES_COL] != best[CLASSES_COL] or baseline[CLASSIFIER_COL] != best[CLASSIFIER_COL]:
        raise SystemExit("the two runs audit different (subset, classifier) configurations: nothing to compare")

    series = [
        (baseline, f"no guidance ($\\alpha = {baseline[ALPHA_COL]:g}$)", SERIES_STYLES[0]),
        (best, f"$\\alpha^{{*}} = {best[ALPHA_COL]:g}$", SERIES_STYLES[1]),
    ]
    values = {row[RUN_ID_COL]: _kldb_values(row[RUN_ID_COL], "synthetic") for row, _, _ in series}
    # The real set does not depend on alpha; either run logs the same one.
    real = _kldb_values(baseline_run_id, "real") if show_real else None

    with mpl.rc_context(paper_rc(FIG_SIZE_IN[0])):
        fig, ax = plt.subplots(figsize=FIG_SIZE_IN, layout="constrained")

        if real is not None:
            _draw_ecdf(ax, real, {**REAL_STYLE, "marker": "none"}, "real ImageNet", with_deciles=False)
        for row, label, style in series:
            _draw_ecdf(ax, values[row[RUN_ID_COL]], {**style, "clip_on": False}, label)

        medians = {row[RUN_ID_COL]: float(np.median(values[row[RUN_ID_COL]])) for row, _, _ in series}
        _annotate_median_shift(ax, medians[baseline_run_id], medians[best_run_id])

        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.01, 0.25))
        ax.set_xlabel("per-image KLDB", color=INK)
        ax.set_ylabel("cumulative fraction of images", color=INK)
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        _style_axes(ax)
        ax.grid(True, axis="x", color=GRIDLINE, linewidth=0.5, zorder=0)

        # Legend handles are built by hand so each entry shows the decile marker
        # the curve itself does not carry. Lower right is the one corner an ECDF
        # leaves empty.
        handles = [
            Line2D([], [], markerfacecolor="white", markeredgewidth=0.8, label=label, **style)
            for _, label, style in series
        ]
        if real is not None:
            handles.append(Line2D([], [], label="real ImageNet", **REAL_STYLE))
        ax.legend(
            handles=handles, loc="lower right", frameon=False, handlelength=2.4, labelcolor=INK, borderaxespad=0.6
        )

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
        stem = FIGURES_DIR / f"kldb_ecdf_alpha0_vs_best_{stamp}"
        fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.01)
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)

    stats = _stats_table(series, values, real)
    caption = _caption(baseline, best, values, real)
    return stem.with_suffix(".png"), stem.with_suffix(".pdf"), caption, stats


def _stats_table(series, values, real):
    """Per-run KLDB quartiles, the validity fraction and the shift against the real set."""
    real_median = None if real is None else float(np.median(real))
    rows = []
    for row, label, _ in series:
        sample = values[row[RUN_ID_COL]]
        rows.append(
            {
                "run": row[RUN_ID_COL],
                "series": label.replace("$", "").replace("\\", ""),
                "alpha": row[ALPHA_COL],
                "n": len(sample),
                "kldb_25": float(np.quantile(sample, 0.25)),
                "kldb_median": float(np.median(sample)),
                "kldb_75": float(np.quantile(sample, 0.75)),
                VALIDITY_COL: row[VALIDITY_COL],
                # How much of the run sits below the real images' median: the
                # share of generated images closer to the boundary than a
                # typical real one.
                "frac_below_real_median": (None if real_median is None else float(np.mean(sample <= real_median))),
            }
        )
    table = pd.DataFrame(rows)
    table.attrs["ks"] = _ks_statistic(*(values[row[RUN_ID_COL]] for row, _, _ in series))
    table.attrs["real_median"] = real_median
    return table


def _caption(baseline, best, values, real):
    """Build the figure caption, with the per-run N spelled out."""
    n_images = int(baseline["num-images"])
    ks = _ks_statistic(values[baseline[RUN_ID_COL]], values[best[RUN_ID_COL]])
    caption = (
        "Empirical CDF of the per-image KLDB for the unguided run "
        f"($\\alpha = {baseline[ALPHA_COL]:g}$) and the selected $\\alpha^{{*}} = {best[ALPHA_COL]:g}$, on "
        f"{_subset_label(best[CLASSES_COL])} under "
        f"{CLASSIFIER_LABELS.get(best[CLASSIFIER_COL], best[CLASSIFIER_COL])}. Lower KLDB is closer to the "
        "decision boundary, so a curve further left is a set of images that probe it harder. Markers give the "
        f"deciles. Each curve is one run over N = {n_images} generated images (seed {int(best['seed'])}); the "
        f"largest vertical gap between the two is {ks:.2f} (two-sample Kolmogorov-Smirnov statistic)."
    )
    if _dominates(values[baseline[RUN_ID_COL]], values[best[RUN_ID_COL]]):
        # Worth stating only because it is checked: no crossing means the gain
        # is not a tail effect traded against the rest of the distribution.
        caption += (
            " The curves do not cross, so guidance moves the whole distribution rather than a tail: at every "
            "KLDB, the guided run has the larger share of images below it."
        )
    if real is None:
        return caption
    return caption + (
        f" The dotted curve is the real ImageNet images of the audited classes under the same classifier "
        f"(N = {len(real)}). The unguided run sits nearly on top of it (KS "
        f"{_ks_statistic(values[baseline[RUN_ID_COL]], real):.2f}): without guidance the generated images are no "
        "closer to the boundary than real ones, and the separation in the figure is what guidance adds."
    )


def main():
    """Draw the figure and print the paths, the per-run quartiles and the caption."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASELINE_RUN_ID, help="run id of the unguided (alpha = 0) run")
    parser.add_argument("--best", default=BEST_RUN_ID, help="run id of the selected alpha* run")
    parser.add_argument("--no-real", action="store_true", help="drop the real ImageNet reference curve")
    args = parser.parse_args()

    png, pdf, caption, stats = plot_kldb_ecdf(args.baseline, args.best, show_real=not args.no_real)
    print(png, pdf, sep="\n")
    print(f"\n{stats.to_string(index=False, float_format='{:.3f}'.format)}")
    print(f"\nKS statistic: {stats.attrs['ks']:.3f}")
    if stats.attrs["real_median"] is not None:
        print(f"real median: {stats.attrs['real_median']:.3f}")
    print(f"\n{caption}")


if __name__ == "__main__":
    main()
