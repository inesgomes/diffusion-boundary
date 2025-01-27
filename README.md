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

## Run

`pyhton -m src --config experiments/<NAME>`

## Available options on HuggingFace

### Datasets

- MINST (28x28)
    - ylecun/mnist
- **CIFAR (32x32)**
    - **uoft-cs/cifar10**
- AFHQ (64x64) - animal faces
    - zzsi/afhq64_16k
- Butterflies (128x18)
    - huggan/smithsonian_butterflies_subset
- CelebAHQ 
    - TODO

### Pre-trained Diffusion Models

- MNIST: 
    - 1aurent/ddpm-mnist
- **CIFAR10**: 
    - **google/ddpm-cifar10-32**
- AFHQ: 
    - krasnova/ddim_afhq_64
- Butterflies: 
    - anton-l/ddpm-butterflies-128
- CelebAHQ: 
    - CompVis/ldm-celebahq-256
    - google/ddpm-celebahq-256

### classifiers

- MINST (28x28)
    - farleyknight-org-username/vit-base-mnist (224x224)
- **CIFAR (32x32)**
    - hf_hub:edadaltocg/resnet50_cifar10
- AFHQ (64x64) - animal faces
    - to train
- Butterflies (128x18)
    - (not possible)
- CelebAHQ 
    - (not easy)


Imagenet bug: https://huggingface.co/datasets/ILSVRC/imagenet-1k/discussions/25/files
