"""
Module that contatins the code to guide the diffusion process with a classifier.

tutorial:
 - https://huggingface.co/learn/diffusion-course/en/unit3/2
 - https://huggingface.co/blog/stable_diffusion
 - https://github.com/huggingface/diffusers/blob/b69fd990ad8026f21893499ab396d969b62bb8cc/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L982
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

    def tensor_to_numpy(self, images):
        """Transform the tensors to numpy."""
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).detach().numpy()
        return images

    @torch.enable_grad()
    def calculate_gradient(self, classifier, transformation, latents, noise_prediction, guidance_type, alpha):
        """Calculate the gradient of the selected metric with respect to the images."""
        # start the gradient calculation
        latents = latents.detach().requires_grad_(True).to(self.device)

        # scale them
        # latent_scaled = self.scheduler.scale_model_input(latents, t)

        # make new noise prediction, only for the original latent and original prompt
        # noise_prediction = self.unet(latents, t, encoder_hidden_states=prompt_embd).sample

        # OLD CODE
        # predict the latent for the current timestep
        # latents_0 = self.scheduler.step(noise_prediction, t, latents).pred_original_sample

        # Save current sigma and compute the original sample according to epsilon prediction type
        sigma_t = self.scheduler.sigmas[self.scheduler.step_index]
        latents_0 = latents - sigma_t * noise_prediction
        # latents_0 = (latents - sigma_t * noise_prediction) / (1 + sigma_t**2).sqrt() # from chatgpt

        # decode the latents to images and transform to the classifier format
        original_lat = 1 / self.vae.config.scaling_factor * latents_0
        images = self.vae.decode(original_lat).sample
        # tensor should be normalized between [0, 1]
        images = images / 2 + 0.5
        images = images - images.min().detach()
        images = images / images.max().detach()
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
        # scaled_gradients = grad / (grad.norm(2).detach() + 1e-8) * latents.norm(2).detach()

        return metric, grad

    def apply_cfg(self, noise_pred_text, noise_pred_uncond, guidance_scale, rescale):
        """Rescale cfg according to https://arxiv.org/pdf/2305.08891."""
        # Apply regular classifier-free guidance.
        noise_prediction = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # Calculate standard deviations.
        std_pos = noise_pred_text.std([1, 2, 3], keepdim=True)
        std_cfg = noise_prediction.std([1, 2, 3], keepdim=True)
        # Apply guidance rescale with fused operations.
        factor = std_pos / std_cfg
        factor = rescale * factor + (1 - rescale)
        return noise_prediction * factor

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
        prompt = kwargs.get("prompt", None)
        guidance_scale = kwargs.get("guidance_scale", 1.0)
        guidance_rescale = kwargs.get("guidance_rescale", 1.0)

        classifier = kwargs.get("classifier", None)
        transformation = kwargs.get("transformation", None)
        alpha = kwargs.get("alpha", None)
        guidance_type = kwargs.get("guidance_type", None)
        guidance_freq = kwargs.get("guidance_freq", 1)

        # this is the height and weight of the original image, not sure if getting this number from the VAE is correct
        height = self.vae.config.sample_size
        width = self.vae.config.sample_size

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

        # TODO: if guidance=1, it means that I don't need to do cfg. So I need to make that implementation

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

        # the k-lms scheduler needs to multiply the latents by its sigma values
        latents = latents * self.scheduler.init_noise_sigma

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents to avoid doing two forward passes
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # 1. predict noise residual
            noise_prediction = self.unet(latent_model_input, t, encoder_hidden_states=text_embd).sample

            # 2. compute text guidance
            noise_pred_uncond, noise_pred_text = noise_prediction.chunk(2)

            # formula from the classifier free guidance
            noise_prediction_cfg = self.apply_cfg(noise_pred_text, noise_pred_uncond, guidance_scale, guidance_rescale)

            # 3. compute classifier guidance (if frequency allows)
            metric = -1
            if (guidance_freq != 0) and (i % guidance_freq == 0):
                metric, grad = self.calculate_gradient(
                    classifier, transformation, latents, noise_pred_text, guidance_type, alpha
                )
                latents = latents.detach() + grad

                # log
                wandb.log({"mean-guidance": metric})
                wandb.log({"loss-guidance": grad})

            # 4. predict the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_prediction_cfg, t, latents).prev_sample

            # log the images over time
            if log_denoising_images & (batch_size == 1):
                latents_ori = 1 / self.vae.config.scaling_factor * latents
                image = self.vae.decode(latents_ori).sample
                image = self.tensor_to_numpy(image)
                image = self.numpy_to_pil(image)
                wandb.log({"denoise_image": wandb.Image(image[0], caption=f"{metric:.4f}"), "diffusion_step": i})

        # deliver the synthetic images
        latents_ori = 1 / self.vae.config.scaling_factor * latents
        images = self.vae.decode(latents_ori).sample

        images = self.tensor_to_numpy(images)
        if output_type == "pil":
            images = self.numpy_to_pil(images)
        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)
