# Stress Testing Classifiers using Diffusion Models

![License](https://img.shields.io/static/v1?label=license&message=CC-BY-NC-ND-4.0&color=green)

Use diffusion models to generate data points close to the decision boundary of classifiers.

## Pre-conditions

- mamba

### env file

Create .env file with the following information
```yaml
CUDA_VISIBLE_DEVICES=0
FILESDIR=<file directory>
ENTITY=<wandb entity to track experiments>
```