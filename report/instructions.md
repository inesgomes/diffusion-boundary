# Report figure instructions

## Figure 3 — coverage + median KLDB

- **Bands:** replace SD with IQR or bootstrap CI of the median — current bands go below zero, impossible for a KL divergence.
- **Y-axis:** log scale, shared across the three panels so subsets are comparable at a glance.
- **Reference line:** horizontal line per panel at the real ImageNet median KLDB, per classifier. Makes "far from the boundary" self-contained.
- **Mark α\*:** a marker or vertical tick per line, so the section's selection is visible in the figure that justifies it.
- **(optional) Per-seed trajectories:** now that pairing is confirmed, faint per-seed lines behind the medians — in the Dogs panel especially, that's what shows the rise past α\* is systematic rather than a shift in the median.
- Annotate N in the caption, and flag it in-panel if the grid ends up heterogeneous.

## Figure 4 — decomposition of KLDB into balance and mass

- 2 rows (ViT-B/16 top, ResNet-50 bottom) × 3 columns (Dogs, Felines, Birds)
- `sharex` within column, `sharey` within row (ViT 0–3, ResNet 0–5); state per-row scaling in the caption
- Column titles on the top row only; x tick labels on the bottom row only
- Row label as the y-axis label: "ViT-B/16 — median value", "ResNet-50 — median value"
- Figure size ≈ 13 × 6 in at 300 dpi for a two-column Springer figure
- x linear on α ∈ [0, 0.105], ticks at the actual grid values
- y linear starting at 0 — the whole point is how close balance gets to zero
- vertical dashed line at α\* (`#5F5E5A`, dashes (3,4)), labelled "α\*" at the top of the axes
- gridlines: horizontal only, `#c3c2b7`, 0.5 pt, alpha 0.45; no vertical grid
- one shared legend below the figure, three entries, horizontal

**Caption:** Median KLDB and its two components against guidance strength α, for each classifier–subset configuration. balance = KL(Unif_C ‖ q) measures how evenly probability is distributed within the audited subset; mass = log(1/m) measures how much probability lies on the subset at all. The identity KLDB = balance + mass holds per image; medians are shown independently and do not sum. Shaded regions are bootstrap 95% confidence intervals of the median over N images. Vertical lines mark the selected α\*. Note the differing y-scales between rows.
