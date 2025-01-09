"""
Module that contatins the code to guide the diffusion process with a classifier.

inspiration: https://huggingface.co/hf-internal-testing/diffusers-dummy-pipeline/blob/main/pipeline.py

tutorial: https://huggingface.co/learn/diffusion-course/en/unit2/2#guidance
"""

from typing import List, Optional, Tuple, Union

import torch
import wandb
from diffusers import DiffusionPipeline, ImagePipelineOutput


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

    def compute_entropy(self, classifier, images):
        """Calculate the entropy of the classifier output."""
        # get classifier output
        probs = classifier.predict(images)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy

    def entropy_guidance(self, classifier, images):
        """Calculate the gradient of the entropy with respect to the images."""
        # Enable gradients for the pixel images
        images = images.clone().detach().requires_grad_(True)

        # Calculate entropy (scalar per image)
        entropy = self.compute_entropy(classifier, images).mean()
        wandb.log({"mean-entropy": entropy})

        # Calculate gradient of the entropy with respect to the images
        entropy_grad = torch.autograd.grad(entropy, images, create_graph=True)[0]
        wandb.log({"loss-entropy": entropy_grad})

        return entropy_grad

    def compute_cd(self, classifier, images):
        """Calculate the confusion distance of the classifier output."""
        # get classifier output
        probs = classifier.predict(images)
        # compute confusion distance (CD) = |0.5 - probs|
        cd = (0.5 - probs).abs()
        return cd

    def cd_guidance(self, classifier, images):
        """Calculate the gradient of the entropy with respect to the images."""
        # Enable gradients for the pixel images
        images = images.clone().detach().requires_grad_(True)

        # Calculate average confusion distance (scalar per image)
        acd = self.compute_cd(classifier, images).mean()
        wandb.log({"acd": acd})

        # Calculate gradient of the confusion distance with respect to the images
        acd_grad = torch.autograd.grad(acd, images, create_graph=True)[0]
        wandb.log({"loss-acd": acd_grad})

        return acd_grad

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

        # TODO: what is eta -> paper from imbalanced data defined eta=0
        # eta = kwargs.get("eta", None)

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
            grad = 0
            if guidance_type == "entropy":
                grad = self.entropy_guidance(classifier, images_0)
            if guidance_type == "acd":
                grad = self.cd_guidance(classifier, images_0)

            images = images.detach() + alpha * grad

            # 3. predict previous mean of image x_t-1 -> do x_t -> x_t-1
            images = self.scheduler.step(noise_prediction, t, images).prev_sample

        # deliver the synthetic images
        images = self.tensor_to_numpy(images)
        if output_type == "pil":
            images = self.numpy_to_pil(images)
        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)
