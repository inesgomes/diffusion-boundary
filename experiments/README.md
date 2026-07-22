# Experiments

This file explains the yaml file with the experiments.


```yaml
project: <wandb project name>

user-args: 
  device: <cuda or cpu>
  seed: <int> # seed for reproducibility
  batch-size: 1 # not yet implemented to generate more than one image at a time
  save-metrics-disk: <boolean> # saves per-image results to $FILESDIR/logs/<run-id>/results_{synthetic,real}.parquet
  save-images-disk: <boolean> # saves the generated images to $FILESDIR/logs/<run-id>/images_{synth,real}.pt
  log-plots: <boolean> # saves plots to wandb 
  display-rgb: True # for RGB images, default=True

dataset:
  name: <name> # dataset name from HuggingFace
  n_classes: <int> # number of classes that the same has
  split: <train or validation or test> 
  num-images: <int> # sampled images from the dataset

classifier:
  name: <name> # classifier name from HuggingFace
  lib: <transformers, timm> # supported libraries to download classifiers from HuggingFace
  calibrate: <boolean> # True to calibrate probabilities (recommended)

diffusion:
  pipeline: latentguidance 
  name: <name> # LDM name from HuggingFace
  type: sd
  args:
    num-inference-steps: <int> # number of timesteps (T)
    prompt-strategy: <classes> # it means that we will use the classes names concatened by "and". Other types of prompts are not yet supported
    classes: <list of class names> # subset of classes that we want to find the decision boundary
    guidance: kl-div-target # KLDB
    alpha: <float or list<float> > # different alpha values to test. 0 means original diffusion without classifier guidance. Unbounded with minimum as zero.
    guidance-freq: <int> # frequency to apply classifier guidance. Between 1 and T. Default=5
    guidance-scale: <float> # it is our beta: guidance scale for classifier free guidance. 1 means to classifier free guidance is applied
    guidance-rescale: <float> # rescaling weight of classifier free guidance. Between 0 and 1 (0 disables it). Required: no default is injected. The experiments use 0.7
    negative-prompt: <empty or text> # negative prompt for the diffusion model

evaluation:
  num-images: <int> # number of images to be generated
  viz-sample-size: <int> # number of images to be visualized in wandb
  attr-map: <boolean> # display attribution map for first generated image
```
