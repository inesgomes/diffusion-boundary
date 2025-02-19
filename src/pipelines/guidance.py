"""
Module that contatins the code to guide the diffusion process with a classifier.

inspiration: https://huggingface.co/hf-internal-testing/diffusers-dummy-pipeline/blob/main/pipeline.py

tutorial: https://huggingface.co/learn/diffusion-course/en/unit2/2#guidance
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
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
    def calculate_gradient(self, classifier, transformation, images, guidance_type, pixel_size):
        """Calculate the gradient of the selected metric with respect to the images."""
        # TODO: change this -> HEAVY (is there a better way?)
        original_n_channels = images.shape[1]

        if transformation is not None:
            images_pil = self.numpy_to_pil(self.tensor_to_numpy(images))
            images = transformation.transform_images(images_pil)

        # Enable gradients for the pixel images
        # images = images.clone().detach().requires_grad_(True).to(self.device)
        images = images.clone().requires_grad_(True).to(self.device)

        # compute the probabilities
        probs, logits = classifier.predict(images)

        # compute the metric
        metric = compute_metric(guidance_type, probs, logits=logits).mean()

        # compute the gradient
        grad = torch.autograd.grad(metric, images)[0]

        # scale gradients for same scale as images
        scaled_gradients = grad / (grad.norm(2).detach() + 1e-8) * images.norm(2).detach()

        # resize if needed
        if scaled_gradients.shape[2] != pixel_size:
            scaled_gradients = F.interpolate(grad, size=(pixel_size, pixel_size), mode="bilinear", align_corners=False)

        # grayscale if needed -> this is a fix to the problem of the classifier outputting 3 channels when the diffuser only uses one channel
        if (original_n_channels == 1) & (scaled_gradients.shape[1] == 3):
            grayscale_gradient = scaled_gradients.mean(dim=1, keepdim=True)
            return metric, grayscale_gradient

        # normal return
        return metric, scaled_gradients

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
            with torch.no_grad():
                noise_prediction = self.unet(images, t).sample

            if (guidance_freq != 0) and (i % guidance_freq == 0):
                # require gradient for the images
                images = images.detach().requires_grad_()

                # prediction of the noise model for the current timestep
                images_0 = self.scheduler.step(noise_prediction, t, images).pred_original_sample

                # 2. compute guidance

                # Get gradient
                metric, grad = self.calculate_gradient(
                    classifier, transformation, images_0, guidance_type, self.unet.config.sample_size
                )
                images = images.detach() + alpha * grad
                # images += alpha * grad

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
