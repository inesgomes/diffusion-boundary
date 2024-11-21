"""
Module that contatins the code to guide the diffusion process with a classifier.

inspiration: https://huggingface.co/hf-internal-testing/diffusers-dummy-pipeline/blob/main/pipeline.py

TODO: implement classifier guidance
"""

import numpy as np
import torch
from diffusers import DiffusionPipeline


class ClassifierGuidance(DiffusionPipeline):
    """Dummy pipeline to test diffusion models with a classifier."""

    def __init__(self, unet, scheduler):
        """_summary_ Constructor for the pipeline.

        Args:
            unet (_type_): _description_ Unet model for diffusion
            scheduler (_type_): _description_ Scheduler for diffusion
        """
        super().__init__()

        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        generator,
        classifier,
        preprocessing,
        num_inference_steps: int,
        batch_size: int = 1,
        arguments=None,
    ):
        """_summary_ Method to guide the diffusion process with a classifier.

        Args:
            generator (_type_): _description_ Random generator for noise
            classifier (_type_): _description_ Classifier to guide the process
            preprocessing (_type_): _description_ Preprocessing image pipeline
            num_inference_steps (int): _description_ Number of inference steps in the scheduler
            alpha (float): _description_ Weight for classifier guidance
            eta (float, optional): _description_. Defaults to 0. Corresponds to η in paper and should be between [0, 1]
            batch_size (int, optional): _description_. Defaults to 1. Number of samples to generate

        Returns:
            _type_: _description_
        """
        # TODO: what is eta -> paper from imbalanced data defined eta=0
        # eta = arguments["eta"]

        # Sample gaussian noise to begin loop
        image = torch.randn(
            (batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size),
            generator=generator,
        )
        image = image.to(self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. classifier prediction
            inputs = preprocessing(images=image, return_tensors="pt")
            outputs = classifier(**inputs)
            prob = outputs.logits.argmax(dim=1)
            print(prob)
            classifier_output = np.abs(np.average(prob) - 0.5)

            # Adjust noise prediction with classifier guidance
            adjusted_output = model_output + arguments["alpha"] * classifier_output

            # 3. predict previous mean of image x_t-1 -> do x_t -> x_t-1
            # add variance depending on eta (eta is only for LDM)
            image = self.scheduler.step(adjusted_output, t, image).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        return image
