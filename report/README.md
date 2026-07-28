# Report figures

Paper figures for the diffusion-boundary experiments, built to stay readable
under colourblind simulation, in greyscale, and at print size.

```bash
python best_alpha_viz.py     # KLDB + coverage over the guidance sweep
python toy_dataset_viz.py    # toy decision boundaries + boundary-proximity metrics
```

Both write PNG + PDF into `figures/` and print the paths. `style.py` holds the
shared palette, typography and print scale — import from it, don't redefine.

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
