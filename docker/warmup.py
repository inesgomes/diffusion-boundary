"""Pre-download all model weights into the image's baked caches at build time.

Runs on CPU (Docker build has no GPU). Mirrors the exact from_pretrained /
ExtractorFactory calls the code uses (src/__main__.py, src/classifier, src/evaluation)
so the runtime cache is hit offline. SD v1-4 is gated -> needs HF_TOKEN (build secret).
"""
import os
import torch

# SECURITY: do NOT call huggingface_hub.login() -- it persists the token to
# HF_HOME/token (and stored_tokens), which would bake the secret into an image
# layer. The HF libraries read HF_TOKEN from the environment automatically for
# gated downloads, and that env var is set ONLY for this build RUN (it is never
# an image ENV), so nothing touches disk.
tok = os.environ.get("HF_TOKEN")
print("HF_TOKEN present in build env:", bool(tok))
if not tok:
    print("WARNING: no HF_TOKEN -> gated SD v1-4 download will fail")

cache = os.environ.get("HF_MODELS_CACHE")

from transformers import (
    CLIPModel, CLIPTokenizer,
    AutoModelForImageClassification, AutoImageProcessor,
)
from diffusers import DiffusionPipeline

print("==> CLIP-L (openai/clip-vit-large-patch14)")
CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

print("==> ViT classifier (google/vit-base-patch16-224)")
AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224", cache_dir=cache)
AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

print("==> Stable Diffusion v1-4 (gated, fp16 variant)")
DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    cache_dir=cache,
    torch_dtype=torch.float16,
    variant="fp16",
)

print("==> eval feature extractors (dino_vits8, inception_fid)")
try:
    from pymdma.image.models.features import ExtractorFactory
    ExtractorFactory.model_from_name(name="dino_vits8")
    ExtractorFactory.model_from_name(name="inception_fid")
    print("    eval extractors cached")
except Exception as e:  # best-effort: if the loader differs, it downloads at first eval
    print("    eval extractor warmup partial/skipped:", repr(e))

print("WARMUP_DONE")
