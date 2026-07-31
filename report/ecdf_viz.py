"""Per-image KLDB ECDFs, unguided against the selected alpha*, one panel per configuration.

The sweep figure reports the median and the IQR per alpha; this one shows the
whole distribution, so a shift that moves the median can be told apart from one
that only moves a tail. Configurations and the alpha* pick come from
wandb-runs.csv, via best_alpha_viz. See README.md for the colour and print-scale
rules.
"""

import argparse
import string
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from best_alpha_viz import (
    ALPHA_COL,
    BASELINE_ALPHA,
    CLASSES_COL,
    CLASSIFIER_COL,
    CLASSIFIER_LABELS,
    FIGURES_DIR,
    RUN_ID_COL,
    SERIES_STYLES,
    VALIDITY_COL,
    _by_config,
    _load_runs,
    _style_axes,
    _subset_label,
    files_dir,
    select_best_alpha,
    subset_sort_key,
)
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from style import GRIDLINE, INK, INK_SECONDARY, TEXT_WIDTH_IN, paper_rc

# The per-image column of results_synthetic.parquet this figure is about: KLDB,
# the metric guidance optimises.
METRIC = "kldb"

# One column per subset, one row per classifier. Both dimensions scale with the
# column count, as in best_alpha_viz: drawing wider is what paper_rc trades for
# smaller type on the page, so the drawn height has to grow in step.
PANEL_WIDTH_IN = 2.3
PANEL_HEIGHT_IN = 1.9

# A guided ECDF dipping below the unguided one by less than this is a handful of
# images swapping order in a tail, not a trade-off worth naming.
CROSSING_TOL = 0.01

# best-of-N is reported as "> CAP" once p falls below 1/CAP: past that the
# estimate rests on a handful of images and the ceiling is noise.
BEST_OF_N_CAP = 100

# Deciles, marked on each curve: markers, not colour, carry identity in print.
DECILES = np.arange(0.1, 0.91, 0.1)

# The real images are a reference, not a series, so they wear ink and a dotted
# line rather than a palette slot - as in the sweep figure.
REAL_STYLE = {"color": INK_SECONDARY, "linestyle": ":"}


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
    """Return F_b - F_a on the pooled sample: the signed vertical gap between two ECDFs."""
    grid = np.sort(np.concatenate([a, b]))
    return np.searchsorted(np.sort(b), grid, "right") / len(b) - np.searchsorted(np.sort(a), grid, "right") / len(a)


def _ks_statistic(a, b):
    """Two-sample Kolmogorov-Smirnov statistic: the largest vertical gap between two ECDFs."""
    return float(np.max(np.abs(_ecdf_gap(a, b))))


def best_of_n(reference, target_median):
    """Express ``target_median`` as a best-of-N draw from ``reference``.

    ``p`` is the fraction of reference images below the target median, so
    ``N = ceil(1 / p)`` is how many reference samples one would draw, keeping
    only the lowest-KLDB one, to match a typical target image. Returns
    ``(p, N)``, with ``N`` None past the cap.
    """
    p = float(np.mean(reference <= target_median))
    return p, (None if p < 1 / BEST_OF_N_CAP else int(np.ceil(1 / p)))


def _n_text(n):
    """Render a best-of-N, capped. The one spelling of it, used by both panel and table."""
    return f"> {BEST_OF_N_CAP}" if n is None else str(n)


def configurations(runs=None, selection=None):
    """Pair each (subset, classifier) with its unguided run and its alpha* run.

    Configurations with no alpha = 0 run are dropped: there is nothing to
    compare them against.
    """
    runs = _load_runs() if runs is None else runs
    picks = _by_config(select_best_alpha(runs) if selection is None else selection)

    pairs = []
    for (classes, classifier), config in runs.groupby([CLASSES_COL, CLASSIFIER_COL], sort=False):
        unguided = config[config[ALPHA_COL] == BASELINE_ALPHA]
        if unguided.empty:
            print(f"no unguided run for {_subset_label(classes)} / {classifier}: panel skipped")
            continue
        pick = picks[(classes, classifier)]
        # The two frames name the run id differently: select_best_alpha renames
        # it to "run_id", the runs frame keeps the exported "ID".
        pairs.append(
            {
                "classes": classes,
                "classifier": classifier,
                "baseline": unguided.iloc[0],
                "best": pick,
                "baseline_id": unguided.iloc[0][RUN_ID_COL],
                "best_id": pick["run_id"],
            }
        )
    return pairs


def load_kldb(pairs, show_real=True):
    """Attach each pair's per-image KLDB arrays, read once and reused by figure and table."""
    for pair in pairs:
        pair["baseline_kldb"] = _kldb_values(pair["baseline_id"], "synthetic")
        pair["best_kldb"] = _kldb_values(pair["best_id"], "synthetic")
        # The real set depends on the subset and the classifier, not on alpha,
        # so either run of the pair logs the same one.
        pair["real_kldb"] = _kldb_values(pair["baseline_id"], "real") if show_real else None
    return pairs


def _panel_stats(pair):
    """Every number one configuration contributes, computed once for the panel, the table and the caption."""
    baseline, best, real = pair["baseline_kldb"], pair["best_kldb"], pair["real_kldb"]
    baseline_median, best_median = float(np.median(baseline)), float(np.median(best))
    # One gap array answers both "how far apart" and "does the guided curve ever
    # fall behind"; a crossing is only worth reporting with its size, since a
    # few images swapping order in a tail is not a trade-off.
    gap = _ecdf_gap(baseline, best)
    # Unguided sampling is the reference: how many standard samples buy one
    # image as close to the boundary as a typical guided one.
    p_ref, n = best_of_n(baseline, best_median)
    return {
        "subset": _subset_label(pair["classes"]).replace("$", ""),
        "classifier": CLASSIFIER_LABELS.get(pair["classifier"], pair["classifier"]),
        "alpha*": pair["best"]["alpha"],
        "n": len(best),
        "kldb_median_0": baseline_median,
        "kldb_median_best": best_median,
        "median_change": best_median / baseline_median - 1,
        "p_ref": p_ref,
        "best_of_n": _n_text(n),
        "ks": float(np.max(np.abs(gap))),
        "worst_deficit": float(max(0.0, -gap.min())),
        "validity_0": pair["baseline"][VALIDITY_COL],
        "validity_best": pair["best"][VALIDITY_COL],
        # How much of each run sits below the real images' median: the share of
        # generated images closer to the boundary than a typical real one.
        "below_real_0": None if real is None else float(np.mean(baseline <= np.median(real))),
        "below_real_best": None if real is None else float(np.mean(best <= np.median(real))),
        "ks_0_vs_real": None if real is None else _ks_statistic(baseline, real),
    }


def _draw_ecdf(ax, values, style, scale, with_deciles=True):
    """One ECDF, drawn as a step curve with its deciles marked."""
    x, y = _ecdf(values)
    # The curve carries no marker of its own: with one step per image it would
    # be 2500 of them. Identity rests on the linestyle and on the deciles below.
    line = {key: value for key, value in style.items() if key != "marker"}
    # where="post": F is right-continuous, so it jumps *at* each observation.
    ax.step(x, y, where="post", linewidth=1.1 * scale, zorder=3, **line)
    if with_deciles:
        ax.plot(
            np.quantile(values, DECILES),
            DECILES,
            marker=style["marker"],
            linestyle="none",
            markersize=3.0 * scale,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=0.8 * scale,
            zorder=4,
        )


def _annotate_panel(ax, stats_row):
    """Name the panel's alpha*, median shift and best-of-N, in the corner an ECDF leaves empty.

    Reads the same row the table and the caption are built from, so the figure
    cannot disagree with its own numbers.
    """
    ax.axhline(0.5, color=INK_SECONDARY, linestyle=(0, (1, 3)), linewidth=0.6, zorder=1)
    # "12" needs the relation spelled out, "> 100" already carries one.
    n_text = stats_row["best_of_n"]
    n_text = f"= {n_text}" if n_text.isdigit() else n_text
    # Top left, not bottom right: both corners are off the curves for an ECDF,
    # but the right-hand one is where the unguided curve climbs once a panel's
    # KLDB runs high, and the block lands on it.
    ax.text(
        0.03,
        0.97,
        f"$\\alpha^{{*}} = {stats_row['alpha*']:g}$"
        f"\nmedian {stats_row['median_change']:+.0%}"
        f"\nbest-of-$N$ {n_text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=INK,
        fontsize="small",
        linespacing=1.3,
        zorder=6,
    )


def plot_kldb_ecdf_grid(show_real=True):
    """Draw one ECDF panel per configuration, save PNG + PDF, and return the paths, caption and stats.

    ``show_real`` adds the real ImageNet images of the audited classes as a
    dotted reference in every panel.
    """
    pairs = load_kldb(configurations(), show_real)

    # Columns: subsets, ordered as in the sweep figure. Rows: classifiers.
    subsets = sorted({pair["classes"] for pair in pairs}, key=subset_sort_key)
    classifiers = sorted({pair["classifier"] for pair in pairs})
    # Computed before drawing so the panel annotations, the printed table and
    # the caption all read the same numbers.
    panels = {(pair["classes"], pair["classifier"]): (pair, _panel_stats(pair)) for pair in pairs}

    fig_size = (PANEL_WIDTH_IN * len(subsets), PANEL_HEIGHT_IN * len(classifiers))
    scale = fig_size[0] / TEXT_WIDTH_IN
    with mpl.rc_context(paper_rc(fig_size[0])):
        fig, axes = plt.subplots(
            len(classifiers),
            len(subsets),
            figsize=fig_size,
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
        fig.set_layout_engine("constrained", w_pad=0.02, h_pad=0.02, hspace=0.04, wspace=0.03)

        for row, classifier in enumerate(classifiers):
            for col, subset in enumerate(subsets):
                ax = axes[row, col]
                panel = panels.get((subset, classifier))
                if panel is None:
                    ax.set_axis_off()
                    continue
                pair, stats_row = panel

                if pair["real_kldb"] is not None:
                    _draw_ecdf(ax, pair["real_kldb"], REAL_STYLE, scale, with_deciles=False)
                for role, style in zip(("baseline_kldb", "best_kldb"), SERIES_STYLES):
                    _draw_ecdf(ax, pair[role], {**style, "clip_on": False}, scale)

                _annotate_panel(ax, stats_row)
                _style_axes(ax)
                ax.grid(True, axis="x", color=GRIDLINE, linewidth=0.5, zorder=0)

                if row == 0:
                    ax.set_title(f"({string.ascii_lowercase[col]}) {_subset_label(subset)}", color=INK, pad=4)
                # The row's classifier, on the outer edge: it is a property of
                # the row, not of any one panel.
                if col == len(subsets) - 1:
                    ax.text(
                        1.04,
                        0.5,
                        CLASSIFIER_LABELS.get(classifier, classifier),
                        transform=ax.transAxes,
                        rotation=270,
                        ha="left",
                        va="center",
                        color=INK,
                    )

        axes[0, 0].set_xscale("log")
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].set_yticks(np.arange(0, 1.01, 0.25))
        axes[0, 0].xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 3.0)))
        axes[0, 0].xaxis.set_minor_formatter(NullFormatter())
        axes[0, 0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axes[0, 0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        # paper_rc has already scaled the rcParam; taking the literal 8.5 instead
        # would silently stop tracking style.py.
        label_size = mpl.rcParams["axes.labelsize"]
        fig.supxlabel("per-image KLDB", fontsize=label_size, color=INK)
        fig.supylabel("cumulative fraction of images", fontsize=label_size, color=INK)

        handles = [
            Line2D(
                [],
                [],
                markerfacecolor="white",
                markeredgewidth=0.8 * scale,
                linewidth=1.1 * scale,
                markersize=3.0 * scale,
                label=label,
                **style,
            )
            for label, style in zip(("no guidance ($\\alpha = 0$)", "selected $\\alpha^{*}$"), SERIES_STYLES)
        ]
        if show_real:
            handles.append(Line2D([], [], linewidth=1.1 * scale, label="real ImageNet", **REAL_STYLE))
        # Anchored below the canvas: constrained layout gives "outside lower
        # center" and the supxlabel the same band, and they collide.
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
        stem = FIGURES_DIR / f"kldb_ecdf_alpha0_vs_best_{stamp}"
        fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.01)
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)

    stats = pd.DataFrame([row for _, row in panels.values()]).sort_values(["classifier", "subset"], ignore_index=True)
    return stem.with_suffix(".png"), stem.with_suffix(".pdf"), _caption(pairs, subsets, stats), stats


def _caption(pairs, subsets, stats):
    """Build the figure caption, with the per-run N spelled out."""
    labels = "; ".join(f"({string.ascii_lowercase[i]}) {_subset_label(subset)}" for i, subset in enumerate(subsets))
    n_images = "/".join(str(n) for n in sorted({int(pair["baseline"]["num-images"]) for pair in pairs}))
    alphas = ", ".join(f"{a:g}" for a in sorted(stats["alpha*"].unique()))
    caption = (
        "Empirical CDF of the per-image KLDB for the unguided run ($\\alpha = 0$) and the selected $\\alpha^{*}$, "
        f"one panel per class subset - {labels} - and per classifier (rows). A curve further left probes the "
        "decision boundary harder; markers give the deciles, the dotted line the median. Each curve is one run "
        f"over N = {n_images} generated images, and the KLDB axis is shared across panels. Selected "
        f"$\\alpha^{{*}}$: {alphas}. Each panel also gives a best-of-$N$ equivalent, which states the guided "
        "median relative to unguided sampling rather than as an absolute KLDB: with $p$ the fraction of "
        f"unguided images below the guided median, $N = \\lceil 1/p \\rceil$ unguided draws - keeping only the "
        f"lowest-KLDB one - match a typical guided image ($N > {BEST_OF_N_CAP}$ when $p < {1 / BEST_OF_N_CAP:g}$)."
    )
    worst = stats["worst_deficit"].max()
    if worst <= CROSSING_TOL:
        caption += (
            " The guided run has the larger share of images below every KLDB, so guidance moves the whole "
            "distribution rather than a tail."
        )
        if worst:
            caption += f" The curves touch only in the extreme tails, by at most {worst:.1%} of the sample."
    else:
        crossing = stats[stats["worst_deficit"] > CROSSING_TOL]
        where = "; ".join(
            f"{row['subset']} / {row['classifier']} ({row['worst_deficit']:.1%})" for _, row in crossing.iterrows()
        )
        caption += (
            f" The curves cross in {len(crossing)} of the {len(stats)} panels ({where}), where the guided run "
            "trails the unguided one over part of the range; elsewhere it leads at every KLDB."
        )
    if stats["ks_0_vs_real"].notna().all():
        caption += (
            " The dotted curve is the real ImageNet images of the audited classes under the same classifier. "
            f"The unguided run sits KS {stats['ks_0_vs_real'].min():.2f} to {stats['ks_0_vs_real'].max():.2f} "
            f"from it, against KS {stats['ks'].min():.2f} to {stats['ks'].max():.2f} between unguided and "
            f"$\\alpha^{{*}}$: guidance moves the generated set off the real distribution by more than the "
            "generator alone departs from it."
        )
    return caption


def main():
    """Draw the grid and print the paths, the per-configuration shift and the caption."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-real", action="store_true", help="drop the real ImageNet reference curves")
    args = parser.parse_args()

    png, pdf, caption, stats = plot_kldb_ecdf_grid(show_real=not args.no_real)
    print(png, pdf, sep="\n")
    print(f"\n{stats.to_string(index=False, float_format='{:.3f}'.format)}")
    print(f"\n{caption}")


if __name__ == "__main__":
    main()
