"""A grid of images sampled from one run's saved images, for visual inspection.

Reads the images a run wrote with ``save-images-disk``, samples a reproducible
random subset of them and pastes it into a single figure. Unlike the other
figures here this one is a photo montage, so it is written as PNG only, at the
images' native resolution: the point is to look at the pixels the run produced.
"""

import argparse
import random
import sys
from pathlib import Path

import torch
from best_alpha_viz import FIGURES_DIR, files_dir
from PIL import Image

KINDS = ("synth", "real")


def images_path(run_id, kind):
    """Return ``FILESDIR/logs/<run_id>/images_<kind>.pt``, where save_images_to_disk writes."""
    return Path(files_dir()) / "logs" / str(run_id) / f"images_{kind}.pt"


def load_run_images(run_id, kind):
    """Load the images ``run_id`` saved for its ``kind`` set."""
    if not files_dir():
        raise SystemExit("FILESDIR is set neither in the environment nor in the project .env")

    path = images_path(run_id, kind)
    if not path.is_file():
        raise SystemExit(f"{path} unreadable: run {run_id} has no {kind} images under FILESDIR")

    with path.open("rb") as file:
        # the file holds pickled PIL images, not tensors, so weights_only cannot be used
        images = torch.load(file, weights_only=False)

    print(f"Loaded {len(images)} {kind} images from {path}")
    return images


def sample_images(images, num_images, seed):
    """Pick ``num_images`` images at random, the same ones for a given seed."""
    if len(images) < num_images:
        print(f"only {len(images)} images available, taking all of them instead of {num_images}", file=sys.stderr)
        num_images = len(images)

    # sorted so the grid reads in run order, which is what the indices below refer to
    indices = sorted(random.Random(seed).sample(range(len(images)), num_images))
    return indices, [images[i] for i in indices]


def build_grid(images, n_cols, cell_size=None, padding=4, background="white"):
    """Paste the images into a grid, keeping their aspect ratio and native resolution.

    The cell defaults to the largest image, so nothing is ever upscaled; the real
    sets hold ImageNet images of mixed sizes, where a fixed ``cell_size`` keeps
    one outlier from setting the scale of the whole figure.
    """
    images = [image.convert("RGB") for image in images]
    if cell_size is None:
        cell_size = max(max(image.size) for image in images)

    n_rows = (len(images) + n_cols - 1) // n_cols
    grid = Image.new(
        "RGB",
        (n_cols * cell_size + (n_cols + 1) * padding, n_rows * cell_size + (n_rows + 1) * padding),
        background,
    )

    for i, image in enumerate(images):
        # only ever downscale, and preserve the aspect ratio when doing so
        if max(image.size) > cell_size:
            image = image.copy()
            image.thumbnail((cell_size, cell_size), Image.LANCZOS)

        row, col = divmod(i, n_cols)
        x = padding + col * (cell_size + padding) + (cell_size - image.width) // 2
        y = padding + row * (cell_size + padding) + (cell_size - image.height) // 2
        grid.paste(image, (x, y))

    return grid


def parse_args():
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id", help="wandb run id, the FILESDIR/logs/<run_id> folder")
    parser.add_argument("seed", type=int, help="seed of the random selection")
    parser.add_argument("--kind", choices=KINDS, default="synth", help="which saved set, default synth")
    parser.add_argument("--num-images", type=int, default=100, help="how many to sample, default 100")
    parser.add_argument("--n-cols", type=int, default=10, help="images per row, default 10")
    parser.add_argument("--cell-size", type=int, help="cell side in pixels, default the largest image's own size")
    parser.add_argument("--padding", type=int, default=4, help="gap between images in pixels, default 4")
    parser.add_argument("--out", type=Path, help="output file, default figures/grid_<kind>_<run_id>_seed<seed>.png")
    return parser.parse_args()


def main():
    """Sample one run's saved images and write them as a single grid figure."""
    args = parse_args()

    images = load_run_images(args.run_id, args.kind)
    indices, selection = sample_images(images, args.num_images, args.seed)
    grid = build_grid(selection, args.n_cols, args.cell_size, args.padding)

    out = args.out or FIGURES_DIR / f"grid_{args.kind}_{args.run_id}_seed{args.seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    # lossless, and compress_level 1 keeps a grid of this size quick to write
    grid.save(out, format="PNG", compress_level=1)

    print(f"sampled indices: {indices}")
    print(f"{len(selection)} images, {grid.width}x{grid.height} -> {out}")


if __name__ == "__main__":
    main()
