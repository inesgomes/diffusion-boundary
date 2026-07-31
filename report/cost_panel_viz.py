"""Benefit and cost of guidance at alpha*, one row per (subset, classifier).

Panel (a) is the median KLDB at alpha* over the same configuration's unguided
median, with real ImageNet-1k and BigGAN on the same scale; panel (b) is FID,
density and coverage at alpha* over the unguided run, oriented so higher is
better, which inverts FID. alpha* comes from best_alpha_viz.select_best_alpha.

See README.md for the colour and print-scale rules.
"""

import string
import sys
from datetime import datetime, timezone

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
    COVERAGE_SOURCE,
    FIGURES_DIR,
    RUN_ID_COL,
    SUBSET_LABELS,
    _class_names,
    _load_runs,
    _style_axes,
    _subset_label,
    real_kldb_reference,
    select_best_alpha,
    subset_sort_key,
)
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from style import CATEGORICAL, GRIDLINE, INK, INK_SECONDARY, TEXT_WIDTH_IN, paper_rc

GUIDED = r"Stress Test Guidance at $\alpha^{*}$"
REAL = "reference ImageNet-1k"
BIGGAN = "BigGAN"

# Median KLDB of BigGAN samples of the audited classes. The one input not in
# wandb-runs.csv, so it is transcribed from the results table and has to be kept
# in sync with it by hand. Emptying it drops the series.
BIGGAN_KLDB = {
    ("Canines", "ViT-B/16"): 3.58,
    ("Canines", "ResNet-50"): 3.58,
    ("Retrievers", "ViT-B/16"): 1.84,
    ("Retrievers", "ResNet-50"): 2.72,
    ("Felines", "ViT-B/16"): 2.41,
    ("Felines", "ResNet-50"): 3.16,
    ("Birds", "ViT-B/16"): 4.22,
    ("Birds", "ResNet-50"): 7.16,
}

# (label, marker, colour, filled), in legend order. Panel (a) has one measured
# quantity and two references, so it wears ink and the secondary neutral and
# leaves the palette whole for panel (b)'s three metrics.
PROXIMITY_SERIES = [(GUIDED, "D", INK, True), (BIGGAN, "o", INK_SECONDARY, False), (REAL, "x", INK_SECONDARY, True)]
QUALITY_SERIES = [
    ("FID", "o", CATEGORICAL[0], True),
    ("density", "s", CATEGORICAL[1], True),
    ("coverage", "^", CATEGORICAL[2], True),
]
INVERTED = {"FID"}  # the one metric a lower raw value is better for
MARKER_SIZE = 3.4

PANEL_WIDTH_IN = 3.0
ROW_HEIGHT_IN = 0.19
CHROME_HEIGHT_IN = 0.95  # titles, x axis, legends: everything that is not a row


def _subset_name(classes_json):
    """Return the subset's bare name, e.g. "Canines" - the key BIGGAN_KLDB is written in."""
    names = _class_names(classes_json)
    return SUBSET_LABELS.get(names, " · ".join(names))


def _picks(runs):
    """Return the alpha* run and the unguided run of each configuration.

    alpha* comes from best_alpha_viz's rule rather than from the run table's
    hand-written Notes, so this figure and the ECDF figure cannot pick different
    runs. A configuration with no unguided run has nothing to be a ratio to.
    """
    best = select_best_alpha(runs).set_index([CLASSES_COL, CLASSIFIER_COL])["run_id"]
    picks = []
    for (classes, classifier), config in runs.groupby([CLASSES_COL, CLASSIFIER_COL], sort=False):
        unguided = config[config[ALPHA_COL] == BASELINE_ALPHA]
        if unguided.empty:
            print(f"{_subset_label(classes)} / {classifier}: no unguided run; skipped", file=sys.stderr)
            continue
        chosen = config[config[RUN_ID_COL] == best[(classes, classifier)]].iloc[0]
        picks.append((classes, classifier, chosen, unguided.iloc[0]))
    return picks


def _real_reference(picks):
    """Median real-image KLDB per configuration, or empty if FILESDIR is unreachable.

    Read off the unguided run, as ecdf_viz and real_baseline.py do: the real set
    does not depend on alpha but is redrawn per run, so the runs of a
    configuration disagree on its median by about a percent.
    """
    selection = pd.DataFrame(
        [{CLASSES_COL: classes, CLASSIFIER_COL: clf, "run_id": base[RUN_ID_COL]} for classes, clf, _, base in picks]
    )
    frame = real_kldb_reference(selection)
    return {} if frame.empty else frame.set_index([CLASSES_COL, CLASSIFIER_COL])["real_kldb_median"].to_dict()


def ratio_table(runs=None):
    """One row per configuration, every value a ratio to the matching unguided run.

    Proximity is normalised per lane, since KLDB depends on the audited
    classifier; quality per subset, since it does not.
    """
    runs = _load_runs() if runs is None else runs
    picks = _picks(runs)
    real = _real_reference(picks)
    unguided = runs[runs[ALPHA_COL] == BASELINE_ALPHA]
    quality_baseline = unguided[unguided[CLASSIFIER_COL] == COVERAGE_SOURCE].set_index(CLASSES_COL)

    rows = []
    for classes, classifier, best, base in picks:
        quality = quality_baseline.loc[classes]
        subset, label = _subset_name(classes), CLASSIFIER_LABELS.get(classifier, classifier)
        row = {
            "classes": classes,
            "subset": subset,
            "classifier": label,
            "run_id": best[RUN_ID_COL],
            "alpha": best[ALPHA_COL],
            "n": int(best["num-images"]),
            GUIDED: best["kldb_median"] / base["kldb_median"],
            REAL: real.get((classes, classifier), np.nan) / base["kldb_median"],
            BIGGAN: BIGGAN_KLDB.get((subset, label), np.nan) / base["kldb_median"],
        }
        for name, *_ in QUALITY_SERIES:
            column = name.lower()
            row[name] = quality[column] / best[column] if name in INVERTED else best[column] / quality[column]
        rows.append(row)

    # Ordered as the other figures order their panels, so a row here and a panel
    # there always mean the same configuration.
    table = pd.DataFrame(rows)
    table["_subset"] = table["classes"].map(subset_sort_key)
    return table.sort_values(["_subset", "classifier"]).drop(columns="_subset").reset_index(drop=True)


def _marker(marker, color, filled, scale):
    """One series' marks, as a kwargs dict the panel and its legend handle share."""
    return {
        "linestyle": "none",
        "marker": marker,
        "markersize": MARKER_SIZE * scale,
        "markerfacecolor": color if filled else "none",
        "markeredgecolor": color,
        "markeredgewidth": 0.8 * scale,
    }


def _draw(ax, table, y, series, span, scale):
    """Draw a panel's marks, its grey span per row, and its legend."""
    for yi, value in zip(y, span):
        ax.plot([value, 1.0], [yi, yi], color=GRIDLINE, linewidth=0.8 * scale, zorder=3)
    # Drawn in order, so a series covers the ones before it where two coincide -
    # except the first, which is the panel's subject and is kept on top.
    for depth, (name, *style) in enumerate(series):
        ax.plot(table[name], y, zorder=5 + (depth == 0), **_marker(*style, scale))
    ax.legend(
        handles=[Line2D([], [], label=name, **_marker(*style, scale)) for name, *style in series],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=len(series),
        frameon=False,
        handlelength=1.4,
        handletextpad=0.4,
        columnspacing=1.2,
        borderpad=0.0,
        labelcolor=INK,
    )


def _frame(ax, table, y, scale):
    """Draw the chrome both panels share: categorical rows, vertical grid, unity marked."""
    ax.axvline(1.0, color=INK_SECONDARY, linestyle=(0, (3, 3)), linewidth=0.6 * scale, zorder=2)
    ax.set_yticks(y)
    ax.set_ylim(-0.7, len(table) - 0.3)
    _style_axes(ax)
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", color=GRIDLINE, linewidth=0.5, zorder=0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    # A hairline between subsets, so the classifier pairs read as pairs.
    for i in range(1, len(table)):
        if table["subset"].iat[i] != table["subset"].iat[i - 1]:
            ax.axhline(y[i] + 0.5, color=GRIDLINE, linewidth=0.5, zorder=1)


def _proximity(ax, table, y, scale):
    """Panel (a). Linear and anchored at 0, as panel (b) is, so a span means the same on both."""
    series = [s for s in PROXIMITY_SERIES if table[s[0]].notna().any()]
    _draw(ax, table, y, series, table[GUIDED], scale)
    upper = max(table[[name for name, *_ in series]].max().max(), 1.0) * 1.08
    ax.set_xlim(0, upper)
    ax.set_xticks(np.arange(0, upper, 0.5))
    ax.set_xlabel("median KLDB (ratio to unguided)", color=INK)


def _quality(ax, table, y, scale):
    """Panel (b)."""
    names = [name for name, *_ in QUALITY_SERIES]
    _draw(ax, table, y, QUALITY_SERIES, table[names].min(axis=1), scale)
    ax.set_xlim(0, max(1.0, table[names].max().max()) * 1.12)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_xlabel("image quality retained (ratio to unguided)", color=INK)


def plot_benefit_cost(table):
    """Draw both panels, save PNG + PDF, and return the paths and the caption."""
    y = np.arange(len(table))[::-1]
    fig_size = (PANEL_WIDTH_IN * 2, ROW_HEIGHT_IN * len(table) + CHROME_HEIGHT_IN)
    scale = fig_size[0] / TEXT_WIDTH_IN

    with mpl.rc_context(paper_rc(fig_size[0])):
        fig, axes = plt.subplots(1, 2, figsize=fig_size, sharey=True, layout="constrained")
        fig.set_layout_engine("constrained", w_pad=0.02, h_pad=0.02, wspace=0.03)

        _proximity(axes[0], table, y, scale)
        _quality(axes[1], table, y, scale)
        for i, (ax, title) in enumerate(zip(axes, ("boundary proximity gained", "image quality retained"))):
            _frame(ax, table, y, scale)
            ax.set_title(f"({string.ascii_lowercase[i]}) {title}", color=INK, pad=4)

        # Set once, on the shared axis: blanking the right-hand panel's labels
        # with set_yticklabels([]) would blank the left-hand panel's too.
        axes[0].set_yticklabels([f"{row.subset} · {row.classifier}" for row in table.itertuples()])
        axes[1].tick_params(labelleft=False)

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        # UTC then back to local: aware, but still reads as local time.
        stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
        stem = FIGURES_DIR / f"benefit_cost_at_best_alpha_{stamp}"
        fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.01)
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)

    return stem.with_suffix(".png"), stem.with_suffix(".pdf"), _caption(table)


def _caption(table):
    """Build the caption from the same table the figure is drawn from."""
    n_images = "/".join(str(n) for n in sorted(table["n"].unique()))
    alphas = ", ".join(f"{a:g}" for a in sorted(table["alpha"].unique()))
    caption = (
        "Benefit and cost of guidance at the selected $\\alpha^{*}$, one row per class subset and classifier. "
        "(a) Median KLDB at $\\alpha^{*}$ as a ratio to the unguided run of the same subset and the same "
        "classifier, so each row is normalised inside its own lane; the grey span runs to 1.0, unguided sampling. "
        "(b) FID, density and coverage at $\\alpha^{*}$ as ratios to the unguided run, oriented so that higher is "
        "better and 1.0 is unchanged, which inverts FID. Image quality does not depend on the audited classifier, "
        f"so both rows of a subset share one unguided run. Each row is one run over N = {n_images} generated "
        f"images; selected $\\alpha^{{*}}$: {alphas}."
    )
    references = [name for name in (BIGGAN, REAL) if table[name].notna().any()]
    if references:
        caption += f" The {' and '.join(references)} marks in (a) are the same ratio for those sources."
    return caption


def main():
    """Draw the figure and print the paths, the per-configuration ratios and the caption."""
    table = ratio_table()
    png, pdf, caption = plot_benefit_cost(table)
    print(png, pdf, sep="\n")

    # The panel (a) series carry their legend labels as column names, and a
    # LaTeX one is unreadable as a terminal header.
    printable = table.rename(columns={GUIDED: "guided", REAL: "real", BIGGAN: "biggan"})
    columns = ["subset", "classifier", "alpha", "run_id", "guided", "real", "biggan"]
    columns += [name for name, *_ in QUALITY_SERIES]
    print(f"\n{printable[columns].to_string(index=False, float_format='{:.3f}'.format)}")
    print(f"\n{caption}")


if __name__ == "__main__":
    main()
