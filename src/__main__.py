"""This is the main file for the diffusion-boundary package."""

import torch

from src.classifier import get_classifier
from src.diffusion import get_custom_pipe


def generate_sample(pipe, classifier, preprocessing, num_inference_steps, alpha, device):
    """Generate a sample image using the pipeline specified."""
    generator = torch.Generator(device=device).manual_seed(42)
    out = pipe(
        generator=generator,
        classifier=classifier,
        preprocessing=preprocessing,
        num_inference_steps=num_inference_steps,
        alpha=alpha,
    )

    return out[0]


def main(num_inference_steps, alpha, device):
    """Generate a sample image."""
    # get classifier and pipe
    pipe = get_custom_pipe(device)
    classifier, preprocessing = get_classifier(device)

    # generate sample and save
    image = generate_sample(pipe, classifier, preprocessing, num_inference_steps, alpha, device)
    image.save("samples/tst.png")


if __name__ == "__main__":
    # TODO: integrate with wandb

    # TODO: receive via args or config file
    # seed
    DEVICE = "cuda:0"
    NUM_INFERENCE_STEPS = 5
    ALPHA = 0

    main(NUM_INFERENCE_STEPS, ALPHA, DEVICE)
