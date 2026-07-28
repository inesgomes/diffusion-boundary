"""Shared palette, typography and print scale for the report figures.

Every figure imports from here rather than defining its own. See README.md for
the palette's CVD validation and the print-scale contract.
"""

# Ordered categorical palette: violet, vermillion, bluish green. Assign by slot,
# never cycle. No fourth slot passes the CVD floor while clearing cividis.
CATEGORICAL = ["#762A83", "#D55E00", "#009E73"]

# Region tints, matching CATEGORICAL slot for slot.
CATEGORICAL_TINTS = ["#F3EDF7", "#FBEDE2", "#E9F5EF"]

# Neutrals for ink and chrome. Text wears these, never a series colour.
INK, INK_SECONDARY, GRIDLINE, AXIS = "#000000", "#333333", "#d9d9d9", "#4d4d4d"

# Full width of a two-column page: what every figure is printed at, whatever
# width it was drawn at.
TEXT_WIDTH_IN = 6.9

PAPER_RC = {
    "font.family": "serif",
    # DejaVu Serif last: matplotlib's own fallback is DejaVu *Sans*, which would
    # silently drop a serif figure back to sans on a machine with no Times.
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,  # TrueType, not Type 3: required by most publishers
    "ps.fonttype": 42,
}

# Everything in PAPER_RC measured in points, and so tied to the drawn size.
_POINT_KEYS = (
    "font.size",
    "axes.titlesize",
    "axes.labelsize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",
    "axes.linewidth",
    "lines.linewidth",
    "lines.markersize",
    "xtick.major.width",
    "ytick.major.width",
)


def paper_rc(fig_width_in=TEXT_WIDTH_IN):
    """PAPER_RC for a figure drawn ``fig_width_in`` wide and printed at text width.

    Scaling the point sizes by the same factor LaTeX will shrink the figure by
    means 8 pt prints at 8 pt whatever width the figure is drawn at.
    """
    scale = fig_width_in / TEXT_WIDTH_IN
    if scale == 1:
        return dict(PAPER_RC)
    return {**PAPER_RC, **{key: PAPER_RC[key] * scale for key in _POINT_KEYS}}
