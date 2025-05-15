# Stress Testing Classifiers around the Decision Boundary with Latent Difusion

![License](https://img.shields.io/static/v1?label=license&message=CC-BY-NC-ND-4.0&color=green)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

Code for the submitted paper "*Stress Testing Classifiers around the Decision Boundary with Latent Difusion*" submitted at BMVC 2025.

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

`pyhton -m src --config experiments/<NAME>`

Experiments are explained in `experiments/README.md`
