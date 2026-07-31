"""Top-|C|-subset percentage of one run, recomputed when the run predates the metric.

``topk-subset`` marks an image whose top-|C| predicted classes all lie inside the
audited subset C, and is aggregated as a fraction over the run. Runs logged
before the metric existed have no such column, but they do store the full ranking
of classes per image in ``sorted_labels``, which is all the indicator needs: the
value is recovered exactly rather than approximated, and no model has to be run
again.

The stored column wins whenever the run has one, so this reports the logged
number for a recent run and the reconstructed one for an old run.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from best_alpha_viz import CLASSES_COL, RUN_ID_COL, RUNS_CSV
from real_baseline import MissingRealLog, real_results_path

METRIC = "topk-subset"
# the ranking of every class per image, descending by probability, written by
# evaluation.prepare_dataset_results for every run
RANKING_COLUMN = "sorted_labels"

KINDS = ("real", "synthetic")


def results_path(run_id, kind):
    """Return ``FILESDIR/logs/<run_id>/results_<kind>.parquet``."""
    # real_results_path already resolves FILESDIR; the two kinds are siblings
    return real_results_path(run_id).with_name(f"results_{kind}.parquet")


def audited_classes(run_id):
    """Return the audited subset C of ``run_id``, from wandb-runs.csv.

    The class names are the dataset's own labels, which is what the ranking in
    ``sorted_labels`` is written in, so they compare directly.
    """
    runs = pd.read_csv(RUNS_CSV)
    rows = runs[runs[RUN_ID_COL] == run_id]
    if rows.empty:
        raise SystemExit(f"run {run_id} is not in {RUNS_CSV}: pass the audited classes with --class")
    return list(json.loads(rows.iloc[0][CLASSES_COL]))


def topk_subset_values(run_id, kind, classes=None, recompute=False):
    """Return the per-image indicator of ``run_id``'s ``kind`` set, and where it came from.

    ``recompute`` ignores a stored column and rebuilds it from the ranking, which
    is how the reconstruction was checked against the runs that have both.
    """
    path = results_path(run_id, kind)
    if not path.is_file():
        raise MissingRealLog(f"{path} unreadable: run {run_id} has no {kind} log under FILESDIR")

    names = pq.read_schema(path).names
    if METRIC in names and not recompute:
        return pd.read_parquet(path, columns=[METRIC])[METRIC].to_numpy(dtype=float), "logged"

    if RANKING_COLUMN not in names:
        raise MissingRealLog(f"{path} has neither a {METRIC} column nor {RANKING_COLUMN}: nothing to recompute from")

    classes = audited_classes(run_id) if classes is None else classes
    # A name absent from the ranking would silently never appear in a top-k and
    # drag the percentage down, so it is an error rather than a zero.
    unknown = [name for name in classes if name not in names]
    if unknown:
        raise SystemExit(f"{path} has no column for {unknown}: these are not classes of the dataset it was logged on")

    ranking = pd.read_parquet(path, columns=[RANKING_COLUMN])[RANKING_COLUMN]
    subset, k = set(classes), len(classes)
    # exactly compute_topk_subset: 1 when the whole top-|C| sits inside C
    return np.array([set(row[:k]).issubset(subset) for row in ranking], dtype=float), "recomputed"


def topk_subset_pct(run_id, kind, classes=None, recompute=False):
    """Return the top-|C|-subset percentage of ``run_id``'s ``kind`` set, its N and its source."""
    values, source = topk_subset_values(run_id, kind, classes, recompute)
    return 100 * float(np.mean(values)), len(values), source


def main():
    """Print the top-|C|-subset percentage of the given run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="wandb run id to report on")
    parser.add_argument("--kind", choices=[*KINDS, "both"], default="both", help="which logged set (default both)")
    parser.add_argument(
        "--class",
        dest="classes",
        action="append",
        help="an audited class name, repeated once per class; needed only for a run absent from wandb-runs.csv",
    )
    parser.add_argument(
        "--recompute", action="store_true", help="rebuild from the ranking even if the run logged the metric"
    )
    args = parser.parse_args()

    kinds = KINDS if args.kind == "both" else (args.kind,)
    reported = False
    for kind in kinds:
        try:
            pct, n_images, source = topk_subset_pct(args.run_id, kind, args.classes, args.recompute)
        except MissingRealLog as error:
            print(f"skipped: {error}", file=sys.stderr)
            continue
        print(f"{kind:<10} topk-subset_pct = {pct:6.3f}   (N = {n_images}, {source})")
        reported = True

    if not reported:
        raise SystemExit(f"run {args.run_id} has no readable log under FILESDIR")


if __name__ == "__main__":
    main()
