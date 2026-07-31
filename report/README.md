# Report figures

Paper figures for the diffusion-boundary experiments, built to stay readable
under colourblind simulation, in greyscale, and at print size.

```bash
python best_alpha_viz.py     # KLDB + coverage over the guidance sweep
python ecdf_viz.py           # per-image KLDB ECDF, unguided vs alpha*
python toy_dataset_viz.py    # toy decision boundaries + boundary-proximity metrics
python real_baseline.py      # real-images reference per unguided run -> real_baseline.csv
python topk_subset.py ID     # top-|C|-subset percentage of one run
```

The three figure scripts write timestamped PNG + PDF into `figures/`, print the
paths, and print the caption to paste into the paper. `style.py` holds the shared
palette, typography and print scale — import from it, don't redefine.

## Running

Run from anywhere: Python puts the script's own directory on the import path, so
the imports between these modules resolve either way. Any environment with
pandas, pyarrow and matplotlib will do — plus scikit-learn for
`toy_dataset_viz.py`, which fits its own classifiers. The project's `dfst` is not
required. Matplotlib is needed even by the two scripts that draw nothing: they
import `best_alpha_viz` for `files_dir` and the subset labels.

Two sources are read:

- `wandb/wandb-runs.csv`, the run table exported from W&B, one row per run.
- `FILESDIR/logs/<run_id>/results_{real,synthetic}.parquet`, the per-image
  metrics. `FILESDIR` is read from the environment or from the project `.env`;
  with it unset or the drive unmounted, the scripts that need it exit with the
  unreadable path rather than a traceback.

`best_alpha_viz.py` and `toy_dataset_viz.py` take no arguments; the other three
are detailed below.

### `ecdf_viz.py`

Needs `FILESDIR`. Takes no run ids: the grid is one column per subset and one row
per classifier, and each panel pairs that configuration's `alpha = 0` run with
the `alpha*` that `best_alpha_viz.select_best_alpha` picks for it. A
configuration with no unguided run is named on stdout and skipped. `--no-real`
drops the real-images reference curves, the only part that reads
`results_real.parquet`. Prints a per-configuration table: median shift,
best-of-`N`, KS, worst ECDF deficit, validity, share below the real median. The
best-of-`N` metric is defined in the caption the script prints.

### `toy_dataset_viz.py`

Reads neither the run table nor the logs — the toy classifiers are fitted in the
script. Its figure names are fixed rather than timestamped
(`figures/toy-dataset-<n>classes-<n>boundaries.{png,pdf}`), so a re-run
overwrites the previous copy.

### `real_baseline.py`

```bash
python real_baseline.py                    # every alpha = 0 run -> real_baseline.csv
python real_baseline.py 2df847i6           # one run, printed, no CSV written
python real_baseline.py --out /tmp/rb.csv
```

Needs `wandb-runs.csv` and `FILESDIR`. Summarises the **real** images of every
`alpha = 0` run — median KLDB, median entropy, top-|C|-subset percentage — one
row per (subset, classifier), since the real set does not depend on alpha. A run
whose parquet is unreachable is named on stderr and skipped, so one missing log
does not cost the table. Note `*.csv` is gitignored, this output included.

### `topk_subset.py`

```bash
python topk_subset.py 4aca1j4v                    # both sets
python topk_subset.py 4aca1j4v --kind synthetic
python topk_subset.py 2df847i6 --recompute
python topk_subset.py 2zp8mws4 \
  --class "leopard, Panthera pardus" \
  --class "jaguar, panther, Panthera onca, Felis onca" \
  --class "cheetah, chetah, Acinonyx jubatus"
```

| argument | meaning |
|---|---|
| `run_id` | required |
| `--kind real\|synthetic\|both` | which logged set, default `both` |
| `--class NAME` | an audited class, repeated once per class; the names contain commas, so a single comma-separated list would be ambiguous |
| `--recompute` | rebuild from the ranking even when the run logged the metric |

Needs `FILESDIR`, and `wandb-runs.csv` unless `--class` is given. Runs logged
before the metric existed have no `topk-subset` column, but they do store the
full per-image class ranking in `sorted_labels`, which is all the indicator
needs, so the value is recovered exactly — the reconstruction agrees with the
logged column on all 94 run/set pairs that have both. Each line says whether it
came from the logged column or was `recomputed`. A class name absent from the
parquet is an error, not a silent zero.

Use `--class` for a run that is not in `wandb-runs.csv`: most of the run
directories under `FILESDIR/logs/` are not.

## Colour

One ordered palette, `style.CATEGORICAL`. Slot 0 is the first series in every
figure, slot 1 the second. Assign by slot; never cycle or reorder.

| slot | hex | name |
|---|---|---|
| 0 | `#762A83` | violet |
| 1 | `#D55E00` | vermillion |
| 2 | `#009E73` | bluish green |

Validated over all pairs on paper white: worst CVD separation ΔE 11.0, all slots
≥ 3:1 contrast. Colour is never the only channel — series also carry a marker
shape or linestyle, so identity survives greyscale.

Two things to preserve if you change it:

- **No blue or yellow.** The toy metric panels sit beside cividis colourbars, so
  a blue or yellow marker reads as a point on that ramp rather than as a class.
- **No fourth slot.** Every fourth hue clear of cividis fails either the CVD
  floor or 3:1 on white. A fourth series wants a facet, not a generated colour.

## Print scale

Figures are printed at `TEXT_WIDTH_IN` (6.9 in) but need not be drawn there —
`paper_rc(drawn_width)` rescales the type so 8 pt prints at 8 pt either way.
**Include them at `\textwidth`**, or that compensation is wrong.

Type is Times-metric serif, with `pdf.fonttype 42` for publishers. Check what
shipped with `pdffonts figures/<name>.pdf`.
