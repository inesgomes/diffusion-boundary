"""This is the main file for the diffusion-boundary package."""

import argparse
import math
import os

import pandas as pd
import torch
import wandb
from diffusers import (
    DDIMPipeline,
    DDPMPipeline,
    DiffusionPipeline,
    LMSDiscreteScheduler,
    PNDMPipeline,
)
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer

from src.classifier.factory import ClassifierFactory
from src.classifier.metrics import UNCERTAINTY_METRICS
from src.dataset.aux import get_tst_dataset_streaming
from src.dataset.factory import DatasetFactory
from src.evaluation import (
    calculate_fid_metric,
    calculate_synthetic_metrics,
    prepare_dataset_results,
    visualize_class_distributions,
    visualize_confusion,
    visualize_metrics_distributions,
    visualize_sample_synthetic_images,
    visualize_top_synthetic_metric,
)
from src.utils import generate_group_name, generate_run_id, load_configurations


def save_images_to_disk(images):
    """Save images to disk."""
    path = os.getenv("FILESDIR") + "/logs/" + wandb.run.id
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/images.pkl", "wb") as f:
        torch.save(images, f)
        print("Images saved at", path)


def create_pipeline(diff_type="ddpm", model="google/ddpm-cifar10-32", pipeline=None, device="cpu"):
    """
    General method to load pre-trained diffusion pipelines.

    Parameters:
        diff_type (str): The diffusion type ("ddpm", "ddim", "pndm", etc.).
        model (str): Pretrained model identifier.
        pipeline (str or None): Custom pipeline file path (if any).
        device (str): Device to load the pipeline on ("cpu" or "cuda").

    Returns:
        DiffusionPipeline: The loaded pipeline.
    """
    pipeline_classes = {
        "ddpm": DDPMPipeline,
        "ddim": DDIMPipeline,
        "pndm": PNDMPipeline,
    }

    # Select the pipeline class or fall back to the generic DiffusionPipeline
    pipeline_class = pipeline_classes.get(diff_type, DiffusionPipeline)

    # Handle custom pipeline logic if provided
    custom_pipeline = f"src/pipelines/{pipeline}.py" if pipeline else None

    if diff_type == "sd":
        # stable diffusion needs clip model
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        # stable diffusion allows float16
        pipe = pipeline_class.from_pretrained(
            model,
            custom_pipeline=custom_pipeline,
            clip_model=clip_model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=os.getenv("HF_MODELS_CACHE"),
        ).to(device)
        # from: https://huggingface.co/docs/diffusers/api/schedulers/ddim
        pipe.scheduler = LMSDiscreteScheduler.from_config(
            pipe.scheduler.config,
            rescale_betas_zero_snr=True,  # create images less noisy but nore blurry
            timestep_spacing="trailing",  # both together creates error
            prediction_type="epsilon",
            use_karras_sigmas=True,  # make sure we are using k-lms version
        )
        pipe.enable_attention_slicing()
        return pipe

    # Load and return the pipeline
    return pipeline_class.from_pretrained(
        model, custom_pipeline=custom_pipeline, cache_dir=os.getenv("HF_MODELS_CACHE")
    ).to(device)


def create_arguments(pipeline_name, classifier, dataset, diffusion_arguments):
    """Get arguments for the diffusion pipeline. Currently only for guidance pipeline."""
    args = {}
    if pipeline_name in ("guidance", "latentguidance"):
        args.update(
            {
                "classifier": classifier,
                "transformation": dataset,
                "guidance_type": diffusion_arguments["guidance"],
                "guidance_freq": diffusion_arguments["guidance-freq"],
                "alpha": diffusion_arguments["alpha"],
            }
        )
    if pipeline_name == "latentguidance":
        # get the index of the classes
        classes_idx = [dataset.get_class_idx(class_name) for class_name in diffusion_arguments["classes"]]
        # the prompt strategy is defined in the yaml, as well as all the classes needed
        classes = f"{' and '.join(diffusion_arguments['classes'])}"
        prompt = diffusion_arguments["prompt-strategy"].replace("<classes>", classes)
        print(">> Prompt: ", prompt)
        args.update(
            {
                "prompt": prompt,
                "labels_idx": classes_idx,
                "guidance_scale": diffusion_arguments["guidance-scale"],
                "guidance_rescale": diffusion_arguments["guidance-rescale"],
                "negative_prompt": diffusion_arguments["negative-prompt"],
            }
        )
    return args


def generate_images(diffusion_settings, classifier, dataset, num_images, batch_size, device):
    """Generate images using the diffusion pipeline described in the config file."""
    # get diffusion pipeline
    pipe = create_pipeline(
        diffusion_settings["type"], diffusion_settings["name"], diffusion_settings["pipeline"], device
    )
    # get arguments for the pipeline
    args = create_arguments(diffusion_settings["pipeline"], classifier, dataset, diffusion_settings["args"])

    # generate images in batches
    num_batches = math.ceil(num_images / batch_size)
    images = []
    generator = torch.Generator()
    # generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(4)] # to generate batches
    for i in tqdm(range(num_batches), desc="Generating images"):
        batch_size_to_use = min(batch_size, num_images - len(images))
        generator.seed()
        log_denoising_images = i == 0
        batch_images = pipe(
            generator=generator,
            num_inference_steps=diffusion_settings["args"]["num-inference-steps"],
            batch_size=batch_size_to_use,
            log_denoising_images=log_denoising_images,
            **args,
        ).images
        images.extend(batch_images)

    print(f"Generated {len(images)} images")
    return images


def stress_test_classifier(
    project_name, group_name, default_configs, dataset_config, classifier_config, diffusion_config, evaluation_config
):
    """Stress test a given classifier by generating images using a diffusion pipeline."""
    # diffusion configurarions in wandb format
    diffusion_config_txt = diffusion_config.copy()
    diffusion_config_txt.update(diffusion_config_txt.pop("args", {}))
    diffusion_config_txt.pop("pipeline", {})

    # init wandb
    wandb.init(
        project=project_name,
        group=group_name,
        job_type=diffusion_config["args"]["guidance"],
        entity=os.getenv("ENTITY"),
        name=generate_run_id(),
        config={
            "num-images": evaluation_config["num-images"],
            "certainty-threshold": evaluation_config["certainty-threshold"],
            "classsifier": classifier_config["name"],
            "diffusion": diffusion_config_txt,
            "log-images": default_configs["log-images"],
        },
    )

    # get classifier specifications
    classifier = None
    if classifier_config is not None:
        classifier = ClassifierFactory.model_from_lib(
            classifier_config["lib"],
            classifier_config["name"],
            default_configs["device"],
        )

    # prepare the original dataset, for evaluation purposes, with the same number of samples as the generated ones
    real_images, real_labels, class_labels = get_tst_dataset_streaming(
        dataset_config["name"],
        dataset_config["split"],
        evaluation_config["num-images"],
        dataset_config["subset"],
    )
    real_dataset = DatasetFactory.dataset_from_lib(
        classifier_config["lib"],
        classifier_config["name"],
        dataset_config["name"],
        dataset_config["n_classes"],
        class_labels,
        real_images,
    )
    real_dataset_res = prepare_dataset_results(
        real_dataset,
        classifier,
        default_configs["batch-size"],
        default_configs["device"],
        evaluation_config["mc-dropout"]["n-samples"],
        evaluation_config["mc-dropout"]["threshold"],
        gt=real_labels,
    )

    torch.cuda.empty_cache()

    # prepare synthetic dataset object
    synth_dataset = DatasetFactory.dataset_from_lib(
        classifier_config["lib"],
        classifier_config["name"],
        dataset_config["name"],
        dataset_config["n_classes"],
        class_labels,
        None,
    )

    # generate images
    images = generate_images(
        diffusion_config,
        classifier,
        synth_dataset,
        evaluation_config["num-images"],
        default_configs["batch-size"],
        default_configs["device"],
    )
    synth_dataset.set_images(images)

    torch.cuda.empty_cache()

    # save to disk, if needed
    if default_configs["log-images"]:
        save_images_to_disk(images)

    # prepare results, from synthetic dataset
    synth_dataset_res = prepare_dataset_results(
        synth_dataset,
        classifier,
        default_configs["batch-size"],
        default_configs["device"],
        evaluation_config["mc-dropout"]["n-samples"],
        evaluation_config["mc-dropout"]["threshold"],
    )

    # EVALUATION of the synthetic dataset
    # log uncertainty metrics
    for unc_metric in UNCERTAINTY_METRICS:
        mean_unc_metric = synth_dataset_res[unc_metric].mean()
        # not all uncertainty metrics are available all the time
        if not math.isnan(mean_unc_metric):
            wandb.log({f"{unc_metric}": mean_unc_metric})
            fig = visualize_top_synthetic_metric(
                synth_dataset, synth_dataset_res, unc_metric, default_configs["display-rgb"]
            )
            wandb.log({f"{unc_metric}_sample": wandb.Image(fig)})

    # from features:

    # quality metrics (Improved precision, Improved Recall, Density and Coverage)
    metrics, features_umap = calculate_synthetic_metrics(
        real_dataset,
        synth_dataset,
        default_configs["batch-size"],
        default_configs["device"],
    )
    wandb.log(metrics)
    if default_configs["log-plots"]:
        wandb.log({"umap": wandb.Image(features_umap)})

    # FID score (calculated seperatly because it needs a different feature extractor)
    fid_value = calculate_fid_metric(
        real_dataset, synth_dataset, default_configs["batch-size"], default_configs["device"]
    )
    wandb.log({"FID_score": fid_value})

    # from probabilities

    # distributions (boxplot): metric and classes
    if default_configs["log-plots"]:

        real_vs_synth = pd.concat([real_dataset_res, synth_dataset_res], keys=["Real", "Synthetic"]).reset_index()
        real_vs_synth = real_vs_synth.rename(columns={"level_0": "keys"}).drop(columns=["level_1"])

        # metric distribution (real vs fake)
        dist_metric = visualize_metrics_distributions(real_vs_synth, dataset_config["n_classes"])
        wandb.log({"dist_metrics": wandb.Image(dist_metric)})

        # visualize label information, if number of classes is small
        # TODO: order by the classes most present in the synthetic dataset, and display top-5

        top_5_classes = synth_dataset_res.groupby("pred").size().sort_values(ascending=False).head(5).index
        real_vs_synth_filter = real_vs_synth[real_vs_synth["pred"].isin(top_5_classes)]

        # class distributions for top classes
        dist_labels = visualize_class_distributions(real_vs_synth_filter, top_n=5)
        wandb.log({"dist_labels": wandb.Image(dist_labels)})

        # ambiguity matrix
        if dataset_config["n_classes"] <= 10:
            viz_pairs, table_confusion = visualize_confusion(
                real_dataset_res,
                synth_dataset_res,
                dataset_config["n_classes"],
                evaluation_config["certainty-threshold"],
            )
            wandb.log({"pairs_cm": wandb.Image(viz_pairs)})
            wandb.log({"_boundaries": wandb.Table(dataframe=table_confusion)})

    # sample: grid of images and respective probs

    # entropy is default metric to sort, if we have no guidance
    sort_metric = diffusion_config["args"]["guidance"] if diffusion_config["pipeline"] == "guidance" else "entropy"
    grid, results = visualize_sample_synthetic_images(
        synth_dataset,
        synth_dataset_res,
        evaluation_config["viz-sample-size"],
        sort_metric,
        default_configs["display-rgb"],
        n_cols=5,
    )
    wandb.log({"sample_grid": wandb.Image(grid)})
    wandb.log({"_sample_probabilities": wandb.Table(dataframe=results)})

    # finish wandb
    wandb.finish()


def main(configuration):
    """Run the stress test per configuration."""
    group_name = generate_group_name(configuration)
    user_configs = configuration["user-args"]
    dataset_config = configuration["dataset"]
    classifier_config = configuration["classifier"]
    evaluation_config = configuration["evaluation"]

    guidance_metric = configuration["diffusion"]["args"]["guidance"]
    alpha = configuration["diffusion"]["args"]["alpha"]
    guidance_freq = configuration["diffusion"]["args"]["guidance-freq"]
    diffusion_config = configuration["diffusion"]

    i = 1
    max_i = len(guidance_metric) * len(alpha) * len(guidance_freq)
    for guidance_metric_value in guidance_metric:
        for alpha_value in alpha:
            for guidance_freq_value in guidance_freq:
                diffusion_config["args"]["guidance"] = guidance_metric_value
                diffusion_config["args"]["alpha"] = alpha_value
                diffusion_config["args"]["guidance-freq"] = guidance_freq_value

                # apply stress testing
                print(f"Starting stress test {i}/{max_i}...")
                stress_test_classifier(
                    configuration["project"],
                    group_name,
                    user_configs,
                    dataset_config,
                    classifier_config,
                    diffusion_config,
                    evaluation_config,
                )
                i += 1


if __name__ == "__main__":
    # load environment variables
    load_dotenv()

    # enable torch32 for faster inference
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Set display options for Pandas
    pd.set_option("display.max_rows", None)  # Show all rows
    pd.set_option("display.max_columns", None)  # Show all columns
    pd.set_option("display.width", None)  # Expand the width to fit the data

    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", required=True, help="Configuration file")
    my_args = parser.parse_args()
    # load information from config file
    config = load_configurations(my_args.config_path)

    # execute code
    main(config)
