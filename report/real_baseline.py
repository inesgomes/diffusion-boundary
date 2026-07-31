"""Boundary-proximity summary of the *real* images logged by the unguided runs.

Reads ``FILESDIR/logs/<run_id>/results_real.parquet`` - the per-image metrics of
the real dataset of the audited classes - and reports the median KLDB, the median
entropy and the top-|C|-subset fraction. These are the reference numbers the
generated sets in the report are compared against.

By default it does this for every alpha = 0 run in ``wandb/wandb-runs.csv`` and
writes the table to CSV: the real set does not depend on alpha, so one unguided
run per (subset, classifier) covers every configuration in the sweep. Pass a run
id to summarise that run alone.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from best_alpha_viz import (
    ALPHA_COL,
    BASELINE_ALPHA,
    CLASSES_COL,
    CLASSIFIER_COL,
    CLASSIFIER_LABELS,
    REPORT_DIR,
    RUN_ID_COL,
    RUNS_CSV,
    _subset_label,
    files_dir,
)

# The per-image columns of results_real.parquet this summary is about. KLDB and
# entropy are distributions, so they are summarised by their median (as the sweep
# table does); topk-subset is a 0/1 indicator, so it is summarised as a fraction -
# the same split evaluation.calculate_evaluation_metrics makes for wandb.
QUANTILE_METRICS = ("kldb", "entropy")
FRACTION_METRICS = ("topk-subset",)

SUMMARY_COLUMNS = [
    "run_id",
    "subset",
    "classifier",
    "n_real",
    "kldb_median",
    "entropy_median",
    "topk-subset_pct",
]

OUTPUT_CSV = REPORT_DIR / "real_baseline.csv"


class MissingRealLog(RuntimeError):
    """A run has no readable ``results_real.parquet`` under FILESDIR."""


def real_results_path(run_id):
    """Return the path of the real-dataset parquet of ``run_id``, inside FILESDIR."""
    return Path(files_dir()) / "logs" / str(run_id) / "results_real.parquet"


def load_real_results(run_id):
    """Load the per-image metrics of the real dataset logged by ``run_id``.

    Only the summarised columns are read: the parquet also holds one column per
    ImageNet class, so reading it whole is 1000 columns for three numbers.
    """
    path = real_results_path(run_id)
    if not path.is_file():
        raise MissingRealLog(f"{path} unreadable: run {run_id} has no real-dataset log under FILESDIR")

    wanted = QUANTILE_METRICS + FRACTION_METRICS
    # the schema alone, so a missing column is reported by name instead of as a
    # pyarrow error, and without paying for a read of the file. Read once: in the
    # comprehension it decodes the 1010-column footer per metric.
    names = pq.read_schema(path).names
    missing = [metric for metric in wanted if metric not in names]
    if missing:
        raise MissingRealLog(f"{path} has no {', '.join(missing)} column(s): it predates those metrics")

    return pd.read_parquet(path, columns=list(wanted))


def summarise_real(run_id):
    """Return the median KLDB, the median entropy and the topk-subset fraction of ``run_id``'s real set.

    A metric that is all-NaN (undefined for the configuration, e.g. topk-subset
    with no audited subset) comes back as NaN rather than silently as 0.
    """
    results = load_real_results(run_id)

    summary = {"run_id": run_id, "n_real": len(results)}
    for metric in QUANTILE_METRICS:
        summary[f"{metric}_median"] = float(results[metric].median())
    for metric in FRACTION_METRICS:
        # sum/N over the images that have the indicator, so an unusable column
        # stays NaN instead of counting as no image in the subset.
        summary[f"{metric}_fraction"] = float(results[metric].mean())
    return summary


def unguided_runs():
    """Return the alpha = 0 rows of wandb-runs.csv, one per (subset, classifier).

    The real set is a property of the audited subset and the classifier, not of
    the generation, so a duplicated configuration is read once.
    """
    df = pd.read_csv(RUNS_CSV)[[RUN_ID_COL, ALPHA_COL, CLASSES_COL, CLASSIFIER_COL]]
    unguided = df[df[ALPHA_COL] == BASELINE_ALPHA].dropna(subset=[RUN_ID_COL])
    return unguided.drop_duplicates(subset=[CLASSES_COL, CLASSIFIER_COL])


def summarise_unguided_runs(runs=None):
    """Summarise the real set of every unguided run, one row per run.

    A run whose log is unreachable is reported on stderr and skipped, so one
    missing parquet does not cost the whole table.
    """
    runs = unguided_runs() if runs is None else runs
    if not files_dir():
        raise SystemExit("FILESDIR is set neither in the environment nor in .env")

    rows = []
    for _, run in runs.iterrows():
        try:
            summary = summarise_real(run[RUN_ID_COL])
        except MissingRealLog as error:
            print(f"skipped: {error}", file=sys.stderr)
            continue
        rows.append(
            {
                **summary,
                "subset": _subset_label(run[CLASSES_COL]),
                "classifier": CLASSIFIER_LABELS.get(run[CLASSIFIER_COL], run[CLASSIFIER_COL]),
                # the table reads as a percentage; summarise_real keeps the raw
                # fraction, so callers stay on wandb's topk-subset_fraction scale
                **{f"{metric}_pct": 100 * summary[f"{metric}_fraction"] for metric in FRACTION_METRICS},
            }
        )

    if not rows:
        raise SystemExit(f"no unguided run in {RUNS_CSV} has a readable real-dataset log")
    return pd.DataFrame(rows)[SUMMARY_COLUMNS].sort_values(["subset", "classifier"], ignore_index=True)


def _print_one(run_id):
    """Print the real-dataset summary of a single run."""
    try:
        summary = summarise_real(run_id)
    except MissingRealLog as error:
        raise SystemExit(error) from error

    print(real_results_path(run_id))
    print(f"\nN = {summary['n_real']} real images")
    print(f"median KLDB:           {summary['kldb_median']:.3f}")
    print(f"median entropy:        {summary['entropy_median']:.3f}")
    print(f"topk-subset fraction:  {summary['topk-subset_fraction']:.2%}")


def main():
    """Summarise the unguided runs into a CSV, or print one run passed on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_id",
        nargs="?",
        help="summarise this run alone and print it, instead of every alpha = 0 run in wandb-runs.csv",
    )
    parser.add_argument("--out", type=Path, default=OUTPUT_CSV, help=f"CSV to write (default {OUTPUT_CSV.name})")
    args = parser.parse_args()

    if args.run_id is not None:
        _print_one(args.run_id)
        return

    table = summarise_unguided_runs()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    # subset labels carry LaTeX maths, as everywhere else in report/, so the CSV
    # drops straight into the paper and the printed copy is the one to read here.
    print(table.to_string(index=False, float_format="{:.3f}".format))
    print(f"\n{len(table)} unguided runs -> {args.out}")


if __name__ == "__main__":
    main()
