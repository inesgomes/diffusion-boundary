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
from diffusers import DDIMScheduler, DiffusionPipeline, ImagePipelineOutput

from src.classifier.metrics import compute_metric


class LatentClassifierGuidance(DiffusionPipeline):
    """Pipeline to test diffusion models with a classifier."""

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
        self.skipped_guidance_updates = 0
        # guidance diagnostics, accumulated over the run and reported once in the summary.
        # per sample, like skipped_guidance_updates, so the skip rate is a real ratio
        self._guidance_updates = 0
        # only the smallest norm matters: the normalization below divides the magnitude out, and
        # a norm that underflows to zero in fp16 makes the update a silent no-op
        self._grad_norm_min = float("inf")
        self._nonfinite_grads = 0
        # per-step logging, enabled only for single-image debug runs (see __call__)
        self._log_guidance_steps = False
        self._last_grad_stats = {}

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
    def calculate_gradient(
        self, classifier, transformation, labels_idx, latents, t, prompt_embd, guidance_type, step=None
    ):
        """Calculate the gradient of the selected metric with respect to the images."""
        # start the gradient calculation
        latents = latents.clone().detach().requires_grad_(True).to(self.device)

        # redo noise prediction, but only for the original latent and original prompt
        latent_scaled = self.scheduler.scale_model_input(latents, t)
        noise_prediction = self.unet(latent_scaled, t, encoder_hidden_states=prompt_embd).sample

        # calculate prediction for the original sample, as parameterized by the scheduler in use
        if isinstance(self.scheduler, DDIMScheduler):
            # DDIM works with alphas: x_0 = (x_t - sqrt(1 - a_t) * eps) / sqrt(a_t)
            alpha_prod_t = self.scheduler.alphas_cumprod[int(t)].to(device=latents.device, dtype=latents.dtype)
            beta_prod_t = 1 - alpha_prod_t
            latents_0 = (latents - beta_prod_t ** (0.5) * noise_prediction) / alpha_prod_t ** (0.5)
        else:
            # k-LMS works with sigmas: x_0 = x_t - sigma_t * eps
            sigma_t = self.scheduler.sigmas[self.scheduler.step_index]
            latents_0 = latents - sigma_t * noise_prediction

        # decode the latents to images and transform to the classifier format
        images = self.decode_latents(latents_0)

        # save images mid denoising
        # wandb.log({"mid_denoise_image": wandb.Image(images), "_diffusion_step": t})

        # classifier transformation
        images_t = transformation.transform_images(images)

        # compute the probabilities and the metric
        probs, logits = classifier.predict(images_t)
        metric = compute_metric(
            guidance_type,
            probs,
            logits=logits,
            labels_idx=labels_idx,
            raw_logits=classifier.raw_logits(logits),
        ).mean()

        # compute the gradient, in relation to the original latent
        grad = torch.autograd.grad(abs(metric), latents)[0]

        # norm gradients to L2 norm (according to https://arxiv.org/pdf/2310.00158)
        # beaware of exploding gradients
        grad_finite = bool(torch.isfinite(grad).all())
        # normalize per sample, over every dimension except the batch, so that samples in a
        # batch are not coupled through a single shared norm
        sample_dims = list(range(1, grad.ndim))
        norm = torch.linalg.vector_norm(grad, dim=sample_dims, keepdim=True).detach()
        # Accumulate the smallest gradient norm and the non-finite count for the run summary.
        grad_norm_log = norm.mean().item() if grad_finite else float("nan")
        self._guidance_updates += grad.shape[0]
        if grad_finite:
            self._grad_norm_min = min(self._grad_norm_min, norm.min().item())
        else:
            self._nonfinite_grads += grad.shape[0]
        self._last_grad_stats = {
            "guidance/grad_norm": grad_norm_log,
            "guidance/nonfinite_grad": int(not grad_finite),
        }
        if grad_finite:
            if not bool((norm > 0).all()):
                # the gradient is fp16, so its norm underflows to 0 below a gradient scale of
                # ~1e-7, and those samples silently keep their (zero) update
                self.skipped_guidance_updates += int((~(norm > 0)).sum())
                print(
                    f"Warning: gradient norm is zero at step {step} (t={int(t)}), skipping the "
                    f"guidance update. Total skipped: {self.skipped_guidance_updates}."
                )
            # no epsilon: a scalar factor in the loss then cancels exactly. Dividing by 1 where the
            # norm is zero leaves that sample's gradient (already all zeros) untouched.
            denominator = torch.where(norm > 0, norm, torch.ones_like(norm))
            latents_norm = torch.linalg.vector_norm(latents, dim=sample_dims, keepdim=True).detach()
            normalized_grad = grad / denominator * latents_norm
        else:
            print("Warning: grad contains NaN or Inf, skipping guidance for this step.")
            normalized_grad = torch.zeros_like(grad)

        # another option: norm gradients to L inf norm (according to https://arxiv.org/pdf/2203.17260) - alpha [0, 1]

        # minimize or maximize?
        if metric < 0:
            return -metric, -normalized_grad
        return metric, normalized_grad

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

    def guidance_step(
        self, latents, t, prompt_emb, step, *, classifier, transformation, labels_idx, guidance_type, alpha
    ):
        """Apply one classifier-guidance update to the latents and return them with the metric.

        The classifier arguments are keyword-only to keep the positional signature small.
        """
        metric, grad = self.calculate_gradient(
            classifier, transformation, labels_idx, latents, t, prompt_emb, guidance_type, step=step
        )
        # no metric means the guidance is undefined at this step, so the latents are left untouched
        if metric is None:
            return latents, None

        # weight the metric with our alpha hyperparameter
        update = grad * alpha

        # single-image debug run only, in one call so every series shares the same wandb step.
        # relative_step is not summarised: it is alpha whenever the update is not skipped.
        if self._log_guidance_steps:
            update_norm = update.norm()
            wandb.log(
                {
                    **self._last_grad_stats,
                    "mean-guidance": float(metric),
                    "guidance/update_norm": update_norm.item(),
                    "guidance/relative_step": (update_norm / latents.norm()).item(),
                    "_diffusion_step": step,
                }
            )
        return latents + update, metric

    def setup_guidance_logging(self, log_denoising_images, batch_size):
        """Enable the per-step guidance scalars, which are only meaningful for a single image.

        Across a batched run they concatenate one trajectory per image into a single series that
        is not a curve of anything, so outside that case only the run summary is reported.
        """
        self._log_guidance_steps = log_denoising_images and (batch_size == 1)
        if self._log_guidance_steps and wandb.run is not None:
            wandb.define_metric("guidance/*", step_metric="_diffusion_step")
            wandb.define_metric("mean-guidance", step_metric="_diffusion_step")

    def guidance_summary(self):
        """Return the run-level guidance diagnostics, or an empty dict if guidance never ran."""
        if self._guidance_updates == 0:
            return {}
        return {
            # denominator for the two counts below, so both read as a rate
            "guidance/updates": self._guidance_updates,
            # gradients whose fp16 norm underflowed to zero, so the update was silently a no-op
            "guidance/skipped_updates": self.skipped_guidance_updates,
            "guidance/nonfinite_grad_count": self._nonfinite_grads,
            # how close the run came to that underflow; inf means every gradient was non-finite
            "guidance/grad_norm_min": self._grad_norm_min,
        }

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
        self.setup_guidance_logging(log_denoising_images, batch_size)

        # stable diffusion
        prompt = kwargs.get("prompt", None)
        negative_prompt = kwargs.get("negative_prompt", "")
        guidance_scale = kwargs.get("guidance_scale", 1.0)
        # 0.0 disables the CFG rescaling (Diffusers convention); the experiments use 0.7 (Lin et al.)
        guidance_rescale = kwargs.get("guidance_rescale", 0.0)
        vae_scale_factor = 8

        # specific for my implementation
        classifier = kwargs.get("classifier", None)
        transformation = kwargs.get("transformation", None)
        labels_idx = kwargs.get("labels_idx", None)
        alpha = kwargs.get("alpha", 0)
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

        # the k-lms scheduler needs to multiply the latents by its sigma values (no-op for DDIM, where it is 1.0)
        latents = latents * self.scheduler.init_noise_sigma

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        metric = None
        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps), start=1):
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

            # 3. compute classifier guidance (if frequency and alpha value allows)
            if (guidance_freq != 0) and (i % guidance_freq == 0) and (alpha > 0):
                latents, metric = self.guidance_step(
                    latents,
                    t,
                    prompt_emb,
                    i,
                    classifier=classifier,
                    transformation=transformation,
                    labels_idx=labels_idx,
                    guidance_type=guidance_type,
                    alpha=alpha,
                )

            # 4. predict the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_prediction, t, latents).prev_sample

            # log the images over time, if only one image is being processed
            if log_denoising_images & (batch_size == 1):
                image = self.decode_latents(latents, output_type)[0]
                caption = f"beta={guidance_scale}"
                if metric is not None:
                    caption = f"{guidance_type}: {metric:.4f}\nalpha={alpha}\n" + caption
                wandb.log({"denoise_image": wandb.Image(image, caption=caption), "_diffusion_step": i})

        # deliver the synthetic images
        images = self.decode_latents(latents, output_type)

        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)
