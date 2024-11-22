# Stress Testing Classifiers using Diffusion Models

![License](https://img.shields.io/static/v1?label=license&message=CC-BY-NC-ND-4.0&color=green)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

Use diffusion models to generate data points close to the decision boundary of classifiers.

## Pre-conditions

- mamba
- pre-commit

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

## Available options

### Pre-trained Diffusion Models

- mnist: 1aurent/ddpm-mnist
- cifar10: google/ddpm-cifar10-32
- butterflies: anton-l/ddpm-butterflies-128
- celebahq: 
    - CompVis/ldm-celebahq-256
    - google/ddpm-celebahq-256


### Datasets
- MINST (28x28)
    - ylecun/mnist
- CIFAR (32x32)
    - uoft-cs/cifar10
- Butterflies (128x18)
    - huggan/smithsonian_butterflies_subset
