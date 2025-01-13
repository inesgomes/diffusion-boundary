"""
Module that contatins the code to guide the diffusion process with a classifier.

inspiration: https://huggingface.co/hf-internal-testing/diffusers-dummy-pipeline/blob/main/pipeline.py

tutorial: https://huggingface.co/learn/diffusion-course/en/unit2/2#guidance
"""

from typing import List, Optional, Tuple, Union

import torch
import wandb
from diffusers import DiffusionPipeline, ImagePipelineOutput

from src.classifier.metrics import (
    compute_confusion_distance,
    compute_entropy,
    compute_norm_entropy,
)


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

    def tensor_to_numpy(self, images):
        """Transform the tensors to numpy."""
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).detach().numpy()
        return images

    def calculate_gradient(self, classifier, images, guidance_type):
        """Calculate the gradient of the selected metric with respect to the images."""
        # Enable gradients for the pixel images
        images = images.clone().detach().requires_grad_(True)
        # compute the probabilities
        probs = classifier.predict(images)

        # compute the metric
        metric = 0
        if guidance_type == "entropy":
            metric = compute_entropy(probs).mean()
        elif guidance_type == "norm_entropy":
            metric = compute_norm_entropy(probs).mean()
        elif guidance_type == "acd":
            metric = compute_confusion_distance(probs).mean()
        else:
            raise ValueError(f"Guidance type {guidance_type} not supported.")

        # compute the gradient
        grad = torch.autograd.grad(metric, images, create_graph=True)[0]

        return metric, grad

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
        guidance_type = kwargs.get("guidance_type", None)

        # Sample gaussian noise to begin loop
        images = torch.randn(
            (batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size),
            generator=generator,
        ).to(self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):

            # 1. predict noise model_output
            with torch.no_grad():
                noise_prediction = self.unet(images, t).sample

            # require gradient for the images
            images = images.detach().requires_grad_()

            # prediction of the noise model for the current timestep
            images_0 = self.scheduler.step(noise_prediction, t, images).pred_original_sample

            # 2. compute guidance

            # Get gradient
            # TODO: implement a new hyperparameter to control the step size
            metric, grad = self.calculate_gradient(classifier, images_0, guidance_type)
            images = images.detach() + alpha * grad

            # 3. predict previous mean of image x_t-1 -> do x_t -> x_t-1
            images = self.scheduler.step(noise_prediction, t, images).prev_sample

            # log
            wandb.log({f"mean-{guidance_type}": metric})
            wandb.log({f"loss-{guidance_type}": grad})

        # deliver the synthetic images
        images = self.tensor_to_numpy(images)
        if output_type == "pil":
            images = self.numpy_to_pil(images)
        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)
