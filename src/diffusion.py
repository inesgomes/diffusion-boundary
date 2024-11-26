"""Module that contains several diffusion pipelines."""

from diffusers import (
    DDIMPipeline,
    DDPMPipeline,
    DiffusionPipeline,
    LDMPipeline,
    PNDMPipeline,
)


def get_default_pipe(diff_type="ddpm", model="google/ddpm-cifar10-32", device="cpu"):
    """Use method to test the DDPM method on CIFAR10 dataset (simple)."""
    # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
    if diff_type == "pndm":
        pipe = PNDMPipeline.from_pretrained(model).to(device)
    elif diff_type == "ddim":
        pipe = DDIMPipeline.from_pretrained(model).to(device)
    elif diff_type == "ddpm":
        pipe = DDPMPipeline.from_pretrained(model).to(device)
    else:
        pipe = DiffusionPipeline.from_pretrained(model).to(device)
    return pipe


def get_custom_pipe(diff_type="ddpm", model="google/ddpm-cifar10-32", pipeline="noguidance", device="cpu"):
    """Use method to load pre-trained models."""
    if diff_type == "pndm":
        pipe = PNDMPipeline.from_pretrained(model, custom_pipeline=f"src/pipelines/{pipeline}.py").to(device)
    elif diff_type == "ddim":
        pipe = DDIMPipeline.from_pretrained(model, custom_pipeline=f"src/pipelines/{pipeline}.py").to(device)
    elif diff_type == "ddpm":
        pipe = DDPMPipeline.from_pretrained(model, custom_pipeline=f"src/pipelines/{pipeline}.py").to(device)
    else:
        pipe = DiffusionPipeline.from_pretrained(model, custom_pipeline=f"src/pipelines/{pipeline}.py").to(device)

    return pipe


def get_ldm_pipe(model="CompVis/ldm-celebahq-256", device="cpu"):
    """Use method to test the ldm pipeline."""
    return LDMPipeline.from_pretrained(model).to(device)
