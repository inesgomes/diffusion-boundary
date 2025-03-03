"""
Module that contatins the code to guide the diffusion process with a classifier.

tutorial:
 - https://huggingface.co/learn/diffusion-course/en/unit3/2
 - https://huggingface.co/blog/stable_diffusion
 - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#
"""

from typing import List, Optional, Tuple, Union

import torch
import wandb
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

    def rescale_noise_cfg(self, noise_cfg, noise_pred_text, guidance_rescale=0.0):
        """Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure.

        Based on Section 3.4 from https://arxiv.org/pdf/2305.08891.pdf.
        Code from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L69
        """
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        return noise_cfg

    @torch.enable_grad()
    def calculate_gradient(self, classifier, transformation, latents, t, prompt_embd, guidance_type, alpha):
        """Calculate the gradient of the selected metric with respect to the images."""
        # start the gradient calculation
        latents = latents.detach().requires_grad_(True).to(self.device)

        # redo noise prediction, but only for the original latent and original prompt
        latent_scaled = self.scheduler.scale_model_input(latents, t)
        noise_prediction = self.unet(latent_scaled, t, encoder_hidden_states=prompt_embd).sample

        # calculate prediction for the original sample
        sigma_t = self.scheduler.sigmas[self.scheduler.step_index]
        latents_0 = latents - sigma_t * noise_prediction

        # decode the latents to images and transform to the classifier format
        images = self.decode_latents(latents_0)

        # classifier transformation
        images_t = transformation.transform_images(images)

        # compute the probabilities and the metric
        probs, logits = classifier.predict(images_t)
        metric = compute_metric(guidance_type, probs, logits=logits).mean()
        # weight the metric with our alpha hyperparameter
        loss = metric * alpha

        # compute the gradient, in relation to the original latent
        grad = torch.autograd.grad(loss, latents)[0]

        # scale gradients
        # scaled_gradients = grad / (grad.norm(2).detach() + 1e-6) * latents.norm(2).detach()

        return metric, grad

    def decode_latents(self, latents, output_type=None):
        """Decode the latents to images. Can be PIL (if explicit) or tensor."""
        # decode VAE
        images = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        # normalize [0,1]
        images = (images / 2 + 0.5).clamp(0, 1)
        if output_type == "pil":
            # to numpy first
            images_np = images.cpu().permute(0, 2, 3, 1).detach().numpy()
            return self.numpy_to_pil(images_np)
        return images

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 100,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        log_denoising_images: bool = False,
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
        # stable diffusion
        prompt = kwargs.get("prompt", None)
        negative_prompt = kwargs.get("negative_prompt", "")
        guidance_scale = kwargs.get("guidance_scale", 1.0)
        guidance_rescale = kwargs.get("guidance_rescale", 1.0)
        vae_scale_factor = 8

        # specific for my implementation
        classifier = kwargs.get("classifier", None)
        transformation = kwargs.get("transformation", None)
        alpha = kwargs.get("alpha", None)
        guidance_type = kwargs.get("guidance_type", None)
        guidance_freq = kwargs.get("guidance_freq", 1)

        # if guidance_scale=1, it means that I don't need to do classifier free guidance
        do_cfg = guidance_scale > 1

        # this is the height and weight of the original image
        height = self.unet.config.sample_size * vae_scale_factor
        width = self.unet.config.sample_size * vae_scale_factor

        # TODO: consider for batch processing
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

        # negative prompt or empty for unconditional embeddings
        if do_cfg:
            max_length = prompt_tok.input_ids.shape[-1]
            uncond_tok = self.tokenizer(
                [negative_prompt], padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_emb = self.text_encoder(uncond_tok.input_ids.to(self.device))[0]
            # classifier-free guidance, needs two forward passes: one with the conditioned input, and another with the unconditional embeddings
            text_emb = torch.cat([uncond_emb, prompt_emb])
        else:
            text_emb = prompt_emb

        # sample latents noise to begin loop
        latents = torch.randn(
            batch_size,
            self.unet.config.in_channels,
            height // vae_scale_factor,
            width // vae_scale_factor,
            generator=generator,
            dtype=text_emb.dtype,
        ).to(self.device)

        # the k-lms scheduler needs to multiply the latents by its sigma values
        latents = latents * self.scheduler.init_noise_sigma

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents to avoid doing two forward passes
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # 1. predict noise residual
            noise_prediction = self.unet(latent_model_input, t, encoder_hidden_states=text_emb).sample

            # 2. compute text guidance
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_prediction.chunk(2)
                # formula from the classifier free guidance
                noise_prediction = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                if guidance_rescale > 0:
                    noise_prediction = self.rescale_noise_cfg(noise_prediction, noise_pred_text, guidance_rescale)

            # 3. compute classifier guidance (if frequency allows)
            metric = -1
            if (guidance_freq != 0) and (i % guidance_freq == 0):
                metric, grad = self.calculate_gradient(
                    classifier, transformation, latents, t, prompt_emb, guidance_type, alpha
                )
                latents = latents.detach() + grad

                # log
                wandb.log({"mean-guidance": metric})
                wandb.log({"loss-guidance": grad})

            # 4. predict the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_prediction, t, latents).prev_sample

            # log the images over time, if only one image is being processed
            if log_denoising_images & (batch_size == 1):
                image = self.decode_latents(latents, output_type)[0]
                wandb.log({"denoise_image": wandb.Image(image, caption=f"{metric:.4f}"), "_diffusion_step": i})

        # deliver the synthetic images
        images = self.decode_latents(latents, output_type)

        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)
