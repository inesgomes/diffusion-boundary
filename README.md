# Stress Testing the Decision Boundaries of Image Classifiers via Latent Difusion

![License](https://img.shields.io/static/v1?label=license&message=CC-BY-NC-ND-4.0&color=green)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

This framework needs:
- **image classifier**: should be a deep learning model
- **dataset**: should be in the same distribution that the classifier was trained - e.g., it can be the training or test set
- **text-to-image latent diffusion model**
- **subset of classes to audit**: should exist in the dataset

Working on Python 3.11.

## Pre-conditions

- mamba
- pre-commit
- Weights & Biases (W&B) account
- HuggingFace Hub account

### create new mamba environent

If working on Linux:

``mamba env create -f environment.yml``

### env file

Create .env file with the following information
```yaml
CUDA_VISIBLE_DEVICES=0
FILESDIR=<file directory>
ENTITY=<wandb entity to track experiments>
PRE_COMMIT_USE_MAMBA=1
HF_HUB_OFFLINE=True
```

### setup W&B and HuggingFace hub

`wandb login`

`huggingface-cli login`

## Run

`python -m src --config experiments/<NAME>`

Experiments are explained in `experiments/README.md`

## BigGAN baseline

To compare against a GAN baseline, the images are generated beforehand with `src/biggan.py`
(`biggan-deep-256`) and saved to `$FILESDIR/biggan/<out>.pt`:

```bash
mkdir -p $FILESDIR/biggan
python src/biggan.py --num_images 2504 --labels 207,208 --trunc 0.4 --batch_size 8 --out dogs_2500
```

`--labels` are ImageNet class ids, repeated cyclically until `num_images`, and must match the
`diffusion.args.classes` of the experiment being compared against.

To evaluate them, set `diffusion.images-path: biggan/dogs_2500.pt` in the experiment (see
`experiments/README.md`) and run it as usual. The generation step is skipped and everything else
works the same way.
