"""
Module that contatins the code to guide the diffusion process with a classifier.

tutorial:https://huggingface.co/learn/diffusion-course/en/unit3/2
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline, ImagePipelineOutput

from src.classifier.metrics import compute_metric


class LatentClassifierGuidance(DiffusionPipeline):
    """Dummy pipeline to test diffusion models with a classifier."""

    def __init__(self, text_encoder, vae, tokenizer, clip_model, feature_extractor, unet, scheduler):
        """Calculate the gradient of the selected metric with respect to the images, for latent diffusion models."""
        super().__init__()

        self.register_modules(
            text_encoder=text_encoder,
            vae=vae,
            tokenizer=tokenizer,
            clip_model=clip_model,
            feature_extractor=feature_extractor,
            unet=unet,
            scheduler=scheduler,
        )

    def tensor_to_numpy(self, images):
        """Transform the tensors to numpy."""
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).detach().numpy()
        return images

    @torch.enable_grad()
    def calculate_gradient(self, classifier, transformation, latents, t, text_embeddings, guidance_type, pixel_size):
        """Calculate the gradient of the selected metric with respect to the images."""
        # prediction of the noise model for the current timestep
        latents = latents.detach().requires_grad_()
        noise_prediction = self.unet(latents, t, encoder_hidden_states=text_embeddings, return_dict=False)[0]

        # TODO: what is going on in here?
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_prediction) / alpha_prod_t ** (0.5)
        fac = torch.sqrt(beta_prod_t)
        images = pred_original_sample * (fac) + latents * (1 - fac)

        # TODO: why not just doing this?
        # latents = 1 / self.vae.config.scaling_factor * latents
        # images = self.vae.decode(latents).sample

        # TODO: change this -> HEAVY (is there a better way?)
        original_n_channels = images.shape[1]  # L vs RGB

        if transformation is not None:
            images_pil = self.numpy_to_pil(self.tensor_to_numpy(images))
            images = transformation.transform_images(images_pil).to(self.device)

        # Enable gradients for the pixel images
        # images = images.clone().detach().requires_grad_(True).to(self.device)
        # images = images.clone().requires_grad_(True).to(self.device)

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
        prompt = kwargs.get("prompt", None)
        guidance_scale = kwargs.get("guidance_scale", 1.0)
        height = kwargs.get("height", 512)
        width = kwargs.get("width", 512)

        # classifier = kwargs.get("classifier", None)
        # transformation = kwargs.get("transformation", None)
        # alpha = kwargs.get("alpha", None)
        # guidance_type = kwargs.get("guidance_type", None)
        # guidance_freq = kwargs.get("guidance_freq", 1)

        # TODO: prepare prompt for multiple images?
        # prompt = [prompt] * batch_size

        # tokenize prompt and get embeddings
        prompt_tok = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_emb = self.text_encoder(prompt_tok.input_ids.to(self.device))[0]

        # unconditional text embeddings for classifier-free guidance, which are just the embeddings for the padding token (empty text)
        max_length = prompt_tok.input_ids.shape[-1]
        uncond_tok = self.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_emb = self.text_encoder(uncond_tok.input_ids.to(self.device))[0]

        # classifier-free guidance, needs two forward passes: one with the conditioned input, and another with the unconditional embeddings
        text_embd = torch.cat([uncond_emb, prompt_emb])

        # sample latents noise to begin loop
        latents = torch.randn(
            batch_size,
            self.unet.config.in_channels,
            height // 8,
            width // 8,
            generator=generator,
            dtype=text_embd.dtype,
        ).to(self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # K-LMS scheduler needs to multiply the latents by its sigma values
        latents = latents * self.scheduler.init_noise_sigma

        for t in self.progress_bar(self.scheduler.timesteps):
            # expand the latents to avoid doing two forward passes
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # 1. predict noise residual
            noise_prediction = self.unet(latent_model_input, t, encoder_hidden_states=text_embd).sample

            # 2. compute text guidance
            noise_pred_uncond, noise_pred_text = noise_prediction.chunk(2)

            # formula from the classifier free guidance
            noise_prediction = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # 3. compute classifier guidance (if frequency allows)
            # if (guidance_freq != 0) and (i % guidance_freq == 0):
            # TODO: why?
            # txt_embd_for_guidance = text_embd.chunk(2)[1]
            # metric, grad = self.calculate_gradient(
            #    classifier, transformation, latents, t, txt_embd_for_guidance, guidance_type, self.unet.config.sample_size
            # )
            # latents = latents.detach() + alpha * grad

            # log
            # wandb.log({"mean-guidance": metric})
            # wandb.log({"loss-guidance": grad})

            # 4. predict the previous noisy sample x_t -> x_t-1
            # latents = self.scheduler.step(noise_prediction, t, latents, **{}, return_dict=False)[0]
            latents = self.scheduler.step(noise_prediction, t, latents).prev_sample

        # deliver the synthetic images
        latents = 1 / self.vae.config.scaling_factor * latents
        images = self.vae.decode(latents).sample

        images = self.tensor_to_numpy(images)
        if output_type == "pil":
            images = self.numpy_to_pil(images)
        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)
