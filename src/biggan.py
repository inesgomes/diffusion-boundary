"""
Generate synthetic images using BigGAN.

Usage:
    python src/biggan.py --num_images 2504 --labels 207,208 --trunc 0.4 --batch_size 8 --out dogs_2500
    python src/biggan.py --num_images 2504 --labels 288,290,293 --trunc 0.4 --batch_size 8 --out felines_2500
    python src/biggan.py --num_images 2504 --labels 139,140,141,142 --trunc 0.4 --batch_size 8 --out birds_2500
"""

import argparse
import os
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from pytorch_pretrained_biggan import (
    BigGAN,
    one_hot_from_int,
    one_hot_from_names,
    truncated_noise_sample,
)


def parse_labels(label_str, num_images):
    """Parse label string into list of class ints.

    Accepts a comma-separated string of integers (ImageNet ids) or WordNet names.
    Returns a list of class-int for each image (length num_images).
    """
    if not label_str:
        # default to random classes sampled uniformly from 0..999
        return list(np.random.randint(0, 1000, size=num_images))
    parts = [p.strip() for p in label_str.split(",") if p.strip() != ""]
    ints = []
    # try parsing as ints
    all_int = True
    for p in parts:
        try:
            ints.append(int(p))
        except ValueError:
            all_int = False
            break
    if all_int:
        if len(ints) == 0:
            return list(np.random.randint(0, 1000, size=num_images))
        # repeat / truncate to match num_images
        reps = (num_images + len(ints) - 1) // len(ints)
        out = (ints * reps)[:num_images]
        return out
    # otherwise treat as names
    # one_hot_from_names returns an array of one-hot vectors; map back to ints
    one_hots = one_hot_from_names(parts)
    # convert to indices
    idxs = [int(np.argmax(v)) for v in one_hots]
    reps = (num_images + len(idxs) - 1) // len(idxs)
    out = (idxs * reps)[:num_images]
    return out


def main():
    """Generate images using BigGAN and save to disk."""
    parser = argparse.ArgumentParser(description="Generate images using BigGAN")
    parser.add_argument("--num_images", type=int, required=True, help="Total number of images to generate")
    parser.add_argument(
        "--labels",
        type=str,
        default="",
        help="Comma-separated ImageNet class ids or class names (WordNet). If fewer than num_images, repeated cyclically.",
    )
    parser.add_argument("--out", type=str, default="images", help="Filename")
    parser.add_argument("--trunc", type=float, default=0.4, help="Truncation psi (0.0 - 1.0)")
    parser.add_argument("--batch_size", type=int, default=8, help="Generation batch size")
    parser.add_argument(
        "--device", type=str, default="cuda:1", help="torch device (e.g. cpu or cuda:0). autodetect if not set"
    )
    args = parser.parse_args()

    device = args.device or ("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    num_images = int(args.num_images)
    labels = parse_labels(args.labels, num_images)

    print(f"Device: {device}, generating {num_images} images, truncation={args.trunc}, batch_size={args.batch_size}")

    # load pretrained BigGAN
    model = BigGAN.from_pretrained("biggan-deep-256")
    model.to(device)
    model.eval()

    all_images = []

    batches = ceil(num_images / args.batch_size)
    for b in range(batches):
        cur_batch = min(args.batch_size, num_images - b * args.batch_size)
        # noise: truncated normal
        noise_np = truncated_noise_sample(truncation=args.trunc, batch_size=cur_batch)
        noise = torch.from_numpy(noise_np).to(device).float()

        # class vectors: one-hot for each label in batch
        batch_labels = labels[b * args.batch_size : b * args.batch_size + cur_batch]
        class_vectors = one_hot_from_int(batch_labels, batch_size=cur_batch)
        class_vectors = torch.from_numpy(class_vectors).to(device).float()

        with torch.no_grad():
            outputs = model(noise, class_vectors, args.trunc)
            outputs = F.interpolate(outputs, size=(224, 224), mode="bilinear", align_corners=False)

            outputs = (outputs + 1) / 2
            outputs = outputs.clamp(0, 1)

            for i in range(cur_batch):
                img = outputs[i]
                img = (img * 255).to(torch.uint8)
                img = img.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
                all_images.append(Image.fromarray(img))

    path = f"{os.getenv('FILESDIR')}/biggan/{args.out}.pt"
    torch.save(all_images, path)
    print(f"Saved compressed {path}")


if __name__ == "__main__":
    load_dotenv()
    main()
