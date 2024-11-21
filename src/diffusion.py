"""Module that contains several diffusion pipelines."""

from diffusers import DDIMPipeline, DDPMPipeline, LDMPipeline, PNDMPipeline


def get_custom_pipe(diff_type="ddpm", model="google/ddpm-cifar10-32", pipeline="noguidance", device="cpu"):
    """
    Use method to load pre-trained models.

    Example:
    butterflies: anton-l/ddpm-butterflies-128
    cifar10: google/ddpm-cifar10-32
    """
    if diff_type == "pndm":
        pipe = PNDMPipeline.from_pretrained(model, custom_pipeline=f"src/pipelines/{pipeline}.py").to(device)
    elif diff_type == "ddim":
        pipe = DDIMPipeline.from_pretrained(model, custom_pipeline=f"src/pipelines/{pipeline}.py").to(device)
    else:
        pipe = DDPMPipeline.from_pretrained(model, custom_pipeline=f"src/pipelines/{pipeline}.py").to(device)

    return pipe


def get_pipe(diff_type="ddpm", model="google/ddpm-cifar10-32", device="cpu"):
    """Use method to test the DDPM method on CIFAR10 dataset (simple)."""
    # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
    if diff_type == "pndm":
        pipe = PNDMPipeline.from_pretrained(model).to(device)
    elif diff_type == "ddim":
        pipe = DDIMPipeline.from_pretrained(model).to(device)
    else:
        pipe = DDPMPipeline.from_pretrained(model).to(device)

    return pipe


def get_ldm_pipe(model="CompVis/ldm-celebahq-256", device="cpu"):
    """Use method to test the ldm pipeline."""
    return LDMPipeline.from_pretrained(model).to(device)
