"""
Module that contatins the code to guide the diffusion process with a classifier.

inspiration: https://huggingface.co/hf-internal-testing/diffusers-dummy-pipeline/blob/main/pipeline.py

tutorial: https://huggingface.co/learn/diffusion-course/en/unit2/2#guidance
"""

from typing import List, Optional, Tuple, Union

import torch
import wandb
from diffusers import DiffusionPipeline, ImagePipelineOutput
from torch.nn import functional as F


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

    def calculate_entropy(self, classifier, images):
        """Calculate the entropy of the classifier output."""
        # get classifier output
        logits = classifier(images)
        probs = F.softmax(logits, dim=1)
        # entropy = F.cross_entropy(logits, probs, reduction='none')
        return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

    def calculate_batch_entropy(self, classifier, images):
        """Calculate the standard deviation of the classifier output, to guide the diffusion process."""
        return self.calculate_entropy(classifier, images).mean()

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
        classifier = kwargs.get("classifier", None)
        alpha = kwargs.get("alpha", None)

        # TODO: what is eta -> paper from imbalanced data defined eta=0
        # eta = kwargs.get("eta", None)

        # Sample gaussian noise to begin loop
        images = torch.randn(
            (batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size),
            generator=generator,
        ).to(self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # TODO: save this to wandb
        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            with torch.no_grad():
                noise_prediction = self.unet(images, t).sample

            # require gradient for the images
            images = images.detach().requires_grad_()
            # prediction of the noise model for the current timestep
            images_0 = self.scheduler.step(noise_prediction, t, images).pred_original_sample

            # 2. compute guidance

            # Calculate loss
            entropy = self.calculate_batch_entropy(classifier, images_0)
            wandb.log({"entropy": entropy})
            loss = entropy * alpha
            wandb.log({"adjusted-loss": loss})

            # Get gradient
            # cond_grad = torch.autograd.grad(loss, images)[0]
            images = images.detach() + torch.autograd.grad(loss, images)[0]

            # 3. predict previous mean of image x_t-1 -> do x_t -> x_t-1
            # add variance depending on eta (eta is only for LDM)
            images = self.scheduler.step(noise_prediction, t, images).prev_sample

        # deliver the synthetic images
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            images = self.numpy_to_pil(images)
        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)
