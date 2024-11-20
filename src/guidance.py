"""
Module that contatins the code to guide the diffusion process with a classifier
"""

import torch
from diffusers import DiffusionPipeline

# inspiration: https://huggingface.co/hf-internal-testing/diffusers-dummy-pipeline/blob/main/pipeline.py


class MyPipeline(DiffusionPipeline):
    """_summary_
    Dummy pipeline to test diffusion models with a classifier
    Args:
        DiffusionPipeline (_type_): _description_
    """
    def __init__(self, unet, scheduler):
        super().__init__()

        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        generator,
        classifier,
        preprocessing,
        num_inference_steps: int,
        alpha: float,
        eta: float = 0,
        batch_size: int = 1,
    ):
        # TODO: what is eta -> paper from imbalanced data defiened eta=0

        # Sample gaussian noise to begin loop
        image = torch.randn(
            (batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size),
            generator=generator,
        )
        print(self.device)
        image = image.to("cuda:0")

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # TODO 2. classifier prediction
            # prob = classifier(preprocess(image).unsqueeze(0))
            # classifier_output = np.abs(prob - 1)
            classifier_output = 0

            # Adjust noise prediction with classifier guidance
            adjusted_output = model_output + alpha * classifier_output

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(adjusted_output, t, image, eta).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        return image
