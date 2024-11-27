"""
Module that contatins the code to guide the diffusion process with a classifier.

inspiration: https://huggingface.co/hf-internal-testing/diffusers-dummy-pipeline/blob/main/pipeline.py
"""

from typing import List, Optional, Tuple, Union

import torch
from diffusers import DiffusionPipeline, ImagePipelineOutput


class NoGuidancePipeline(DiffusionPipeline):
    """Dummy pipeline to unconditional generation with diffusion models without guidance."""

    def __init__(self, unet, scheduler):
        """_summary_ Constructor for the pipeline.

        Args:
            unet (_type_): _description_ Unet model for diffusion
            scheduler (_type_): _description_ Scheduler for diffusion
        """
        super().__init__()

        self.register_modules(unet=unet, scheduler=scheduler)

    def transform_images(self, images):
        """Transform the images to the correct format for visualization."""
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        return images

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 100,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """_summary_ Method to guide the diffusion process with a classifier.

        Args:
            generator (_type_): _description_ Random generator for noise
            num_inference_steps (int): _description_ Number of inference steps in the scheduler
            batch_size (int, optional): _description_. Defaults to 1. Number of samples to generate

        Returns:
            _type_: _description_
        """
        # Sample gaussian noise to begin loop
        images = torch.randn(
            (batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size),
            generator=generator,
        ).to(self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            noise_prediction = self.unet(images, t).sample

            # 2. predict previous mean of image x_t-1 -> do x_t -> x_t-1
            images = self.scheduler.step(noise_prediction, t, images).prev_sample

        # deliver the synthetic images
        images = self.transform_images(images)
        if output_type == "pil":
            images = self.numpy_to_pil(images)
        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)
