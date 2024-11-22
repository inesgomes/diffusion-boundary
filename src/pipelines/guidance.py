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
        num_inference_steps: int,
        batch_size: int = 1,
        **kwargs,
    ):
        """_summary_ Method to guide the diffusion process with a classifier.

        Args:
            generator (_type_): _description_ Random generator for noise
            num_inference_steps (int): _description_ Number of inference steps in the scheduler
            batch_size (int, optional): _description_. Defaults to 1. Number of samples to generate

        Returns:
            _type_: _description_
        """
        classifier = kwargs.get("classifier", None)
        preprocessing = kwargs.get("preprocessing", None)
        alpha = kwargs.get("alpha", None)

        # TODO: what is eta -> paper from imbalanced data defined eta=0
        # eta = kwargs.get("eta", None)

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

            # todo: I think that the model ouput from unet is different from the classifier output. Check what the other girl did

            # 2. classifier prediction
            inputs = preprocessing(images=image, return_tensors="pt")
            outputs = classifier(**inputs)
            classifier_output = np.abs(np.average(outputs.logits.argmax(dim=1)) - 0.5)

            # Adjust noise prediction with classifier guidance
            adjusted_output = model_output + alpha * classifier_output

            # 3. predict previous mean of image x_t-1 -> do x_t -> x_t-1
            # add variance depending on eta (eta is only for LDM)
            image = self.scheduler.step(adjusted_output, t, image).prev_sample

        # TODO: this is assuming only one image -> refactor to handle multiple images
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        return image
