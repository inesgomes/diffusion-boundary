"""This is the main file for the diffusion-boundary package."""

import argparse
import json
import os

import torch
import wandb
from dotenv import load_dotenv
from torch.nn import functional as F
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention

from src.classifier import get_classifier
from src.dataset import get_labels
from src.diffusion import get_custom_pipe, get_pipe
from src.utils import create_image_grid, generate_run_id, load_configurations


def evaluate_classifier(classifier, classifier_type, preprocess, dataset_name, images, device):
    """Evaluate the classifier on the generated images."""
    # 1 vs 3 channels
    if images.shape[3] == 1:
        images = torch.tensor(images).squeeze(-1).unsqueeze(1)
    elif images.shape[3] == 3:
        images = torch.tensor(images).permute(0, 3, 1, 2)  # Rearrange to (batch_size, 3, height, width)
    else:
        raise ValueError(f"Unsupported image shape: {images.shape}")

    # Convert images back to PyTorch tensors
    # TODO: I need to fix this. The preprocessing should be the same
    if classifier_type == "transformers":
        inputs = preprocess(images=images, return_tensors="pt").to(device)
        # get predictions
        outputs = classifier(**inputs)
    else:
        # Normalize images for the classifier
        inputs = preprocess(images).to(device)
        outputs = classifier(inputs)

    # problem: mnist uses this model
    if isinstance(outputs, ImageClassifierOutputWithNoAttention):
        outputs = outputs.logits

    print(outputs)
    print("----------------")

    probabilities = F.softmax(outputs, dim=1)
    top_probs, top_indices = probabilities.topk(10, dim=1)

    labels = get_labels(dataset_name)
    # get labels
    return {
        i: {labels[int(idx)]: round(prob.item(), 2) for idx, prob in zip(top_indices[i], top_probs[i])}
        for i in range(top_indices.size(0))
    }


def main(configuration):
    """Generate a sample image."""
    diffusion_settings = configuration["diffusion"]
    device = configuration["device"]
    group_name = configuration["dataset"]["name"] + "_" + configuration["diffusion"]["pipeline"]

    # init wandb
    wandb.init(
        project=configuration["project"],
        group=group_name,
        job_type=diffusion_settings["type"],
        entity=os.getenv("ENTITY"),
        name=generate_run_id(),
        config={
            "seed": configuration["seed"],
            "diffusion": diffusion_settings,
            "classsifier": configuration["classifier"]["name"],
        },
    )

    # TODO: i dont like this, maybe I need to refactor the code
    args = {}
    classifier = None
    preprocessing = None
    if diffusion_settings["pipeline"] != "default":
        # get custom pipeline -> is only not needed for default pipeline
        pipe = get_custom_pipe(
            diffusion_settings["type"], diffusion_settings["name"], diffusion_settings["pipeline"], device
        )
        if configuration["classifier"] is not None:
            # get classifier
            classifier, preprocessing = get_classifier(
                configuration["classifier"]["lib"], configuration["classifier"]["name"], device
            )
        if diffusion_settings["pipeline"] == "guidance":
            # update pipeline for classifier guidance
            args = {
                "classifier": classifier,
                "preprocessing": preprocessing,
                "alpha": diffusion_settings["args"]["alpha"],
            }
    else:
        # get default pipeline
        pipe = get_pipe(diffusion_settings["type"], diffusion_settings["name"], device)

    # create generator
    generator = torch.Generator(device=device).manual_seed(configuration["seed"])
    # generate images
    out = pipe(
        generator=generator,
        num_inference_steps=diffusion_settings["args"]["num-inference-steps"],
        batch_size=diffusion_settings["args"]["batch-size"],
        **args,
    )

    # Create a grid from the batch for visualization, and save
    grid = create_image_grid(out, max_columns=10)
    wandb.log({"sample_grid": wandb.Image(grid)})

    # evaluate the synthetic images with the classifier (if available)
    if (classifier is not None) and (preprocessing is not None):
        results = evaluate_classifier(
            classifier, configuration["classifier"]["lib"], preprocessing, configuration["dataset"]["name"], out, device
        )
        print("RESULTS:", json.dumps(results, indent=4))
        wandb.log({"results": results})

    # finish wandb
    wandb.finish()


if __name__ == "__main__":
    # TODO: method should be able to receive my trained classifier
    # load environment variables
    load_dotenv()

    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", required=True, help="Configuration file")
    my_args = parser.parse_args()
    # load information from config file
    config = load_configurations(my_args.config_path)

    # execute code
    main(config)
