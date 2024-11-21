"""Module that contains several diffusion pipelines."""

from diffusers import LDMPipeline, PNDMPipeline


def get_custom_pipe(device):
    """
    Use method to load pre-trained models.

    Example:
    butterflies: anton-l/ddpm-butterflies-128
    cifar10: google/ddpm-cifar10-32
    """
    return PNDMPipeline.from_pretrained("google/ddpm-cifar10-32", custom_pipeline="src/guidance.py").to(device)


def get_ldm_pipe(device):
    """Use method to test the ldm pipeline."""
    return LDMPipeline.from_pretrained("CompVis/ldm-celebahq-256").to(device)


def get_ddpm_cifar_pipe(device):
    """Use method to test the DDPM method on CIFAR10 dataset (simple)."""
    # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
    return PNDMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
