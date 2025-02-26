"""
Module that contatins the code to guide the diffusion process with a classifier.

inspiration: https://huggingface.co/hf-internal-testing/diffusers-dummy-pipeline/blob/main/pipeline.py

tutorial: https://huggingface.co/learn/diffusion-course/en/unit2/2#guidance
"""

from typing import List, Optional, Tuple, Union

import torch
import wandb
from diffusers import DiffusionPipeline, ImagePipelineOutput

from src.classifier.metrics import compute_metric


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

    @torch.enable_grad()
    def calculate_gradient(self, classifier, transformation, images, noise_prediction, t, guidance_type, alpha):
        """Calculate the gradient of the selected metric with respect to the images."""
        # require gradient for the images
        images = images.detach().requires_grad_(True).to(self.device)

        # prediction of the noise model for the current timestep
        images_0 = self.scheduler.step(noise_prediction, t, images).pred_original_sample

        # transform according to the classifier
        images_t = transformation.transform_images(images_0)

        # compute the probabilities
        probs, logits = classifier.predict(images_t)

        # compute the metric
        metric = compute_metric(guidance_type, probs, logits=logits).mean()
        loss = metric * alpha

        # compute the gradient
        grad = torch.autograd.grad(loss, images)[0]

        return metric, grad

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
        classifier = kwargs.get("classifier", None)
        transformation = kwargs.get("transformation", None)
        alpha = kwargs.get("alpha", None)
        guidance_type = kwargs.get("guidance_type", None)
        guidance_freq = kwargs.get("guidance_freq", 1)

        # Sample gaussian noise to begin loop
        images = torch.randn(
            (batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size),
            generator=generator,
        ).to(self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):

            # 1. predict noise model_output
            noise_prediction = self.unet(images, t).sample

            if (guidance_freq != 0) and (i % guidance_freq == 0):
                # 2. compute guidance

                # Get gradient
                metric, grad = self.calculate_gradient(
                    classifier, transformation, images, noise_prediction, t, guidance_type, alpha
                )
                images = images.detach() + grad

                # log
                wandb.log({"mean-guidance": metric})
                wandb.log({"loss-guidance": grad})

            # 3. predict previous mean of image x_t-1 -> do x_t -> x_t-1
            images = self.scheduler.step(noise_prediction, t, images).prev_sample

        # deliver the synthetic images
        images = self.tensor_to_numpy(images)
        if output_type == "pil":
            images = self.numpy_to_pil(images)
        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)
