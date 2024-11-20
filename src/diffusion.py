"""
Module that contains several diffusion pipelines
"""

from diffusers import LDMPipeline, PNDMPipeline


def get_custom_pipe(device):
    """_summary_
    use method that load pre-trained models

    butterflies: anton-l/ddpm-butterflies-128
    """
    pipe = PNDMPipeline.from_pretrained("google/ddpm-cifar10-32", custom_pipeline="src/guidance.py")
    pipe.to(device)
    return pipe


def get_ldm_pipe(device):
    """_summary_
    method to test the ldm pipeline
    """
    return LDMPipeline.from_pretrained("CompVis/ldm-celebahq-256").to(device)


def get_ddpm_cifar_pipe(device):
    """_summary_
    Method to test the DDPM method on CIFAR10 dataset (simple)
    """
    # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
    return PNDMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
