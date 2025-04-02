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
from src.classifier.metrics import (
    MULTICLASS_METRICS,
    UNCERTAINTY_METRICS,
)
from src.dataset.aux import get_tst_dataset_streaming
from src.dataset.factory import DatasetFactory
from src.evaluation import (
    calculate_evaluation_metrics,
    calculate_feature_metrics,
    calculate_fid_metric,
    prepare_dataset_results,
)
from src.utils import generate_group_name, generate_run_id, load_configurations
from src.visualization import (
    visualize_class_distributions,
    visualize_confusion,
    visualize_features_umap,
    visualize_metrics_distributions,
    visualize_sample_synthetic_images,
    visualize_top_synthetic_metric,
)


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
        5000,  # evaluation_config["num-images"], # TODO: I need to solve the problem of imbalanced data
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
    real_dataset.set_labels(real_labels)

    real_dataset_res = prepare_dataset_results(
        real_dataset,
        classifier,
        default_configs["batch-size"],
        default_configs["device"],
        evaluation_config["mc-dropout"]["n-samples"],
        evaluation_config["mc-dropout"]["threshold"],
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

    # compute metrics that need the classifer
    synth_dataset_res = prepare_dataset_results(
        synth_dataset,
        classifier,
        default_configs["batch-size"],
        default_configs["device"],
        evaluation_config["mc-dropout"]["n-samples"],
        evaluation_config["mc-dropout"]["threshold"],
    )

    # compute metrics that need features and target (currently is only KDN)
    synth_metrics, real_features, fake_features = calculate_feature_metrics(
        real_dataset,
        synth_dataset,
        diffusion_config["args"]["classes"],
        default_configs["batch-size"],
        default_configs["device"],
    )
    synth_dataset_res = pd.concat([synth_dataset_res, synth_metrics], axis=1)

    # evaluation of dataset
    eval_metrics_name = list(set(UNCERTAINTY_METRICS) | set(MULTICLASS_METRICS) | set(synth_metrics.columns))
    synth_dataset_metrics = synth_dataset_res[eval_metrics_name]
    eval_metrics = calculate_evaluation_metrics(real_features, fake_features, synth_dataset_metrics)

    # fid needs to be calulated in a different way
    fid_value = calculate_fid_metric(
        real_dataset, synth_dataset, default_configs["batch-size"], default_configs["device"]
    )
    eval_metrics["fid"] = fid_value

    # log metrics
    wandb.log(eval_metrics)

    if default_configs["log-plots"]:
        # umap visualization of features
        features_umap = visualize_features_umap(real_features, fake_features)
        wandb.log({"umap": wandb.Image(features_umap)})

        # images with top entropy
        fig = visualize_top_synthetic_metric(
            synth_dataset, synth_dataset_res, sort_metric="entropy", top_n=5, display_rgb=default_configs["display-rgb"]
        )
        wandb.log({"entropy_sample": wandb.Image(fig)})

        # metric distribution (real vs fake) - boxplot
        real_vs_synth = pd.concat([real_dataset_res, synth_dataset_res], keys=["Real", "Synthetic"]).reset_index()
        real_vs_synth = real_vs_synth.rename(columns={"level_0": "keys"}).drop(columns=["level_1"])
        dist_metric = visualize_metrics_distributions(real_vs_synth, dataset_config["n_classes"])
        wandb.log({"dist_metrics": wandb.Image(dist_metric)})

        # class distributions for top classes - boxplot
        # TODO: order by the classes most present in the synthetic dataset, and display top-5
        top_5_classes = synth_dataset_res.groupby("pred").size().sort_values(ascending=False).head(5).index
        real_vs_synth_filter = real_vs_synth[real_vs_synth["pred"].isin(top_5_classes)]
        dist_labels = visualize_class_distributions(real_vs_synth_filter, top_n=5)
        wandb.log({"dist_labels": wandb.Image(dist_labels)})

        # ambiguity matrix (only if low number of classes)
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
    grid, _ = visualize_sample_synthetic_images(
        synth_dataset,
        synth_dataset_res,
        evaluation_config["viz-sample-size"],
        sort_metric,
        default_configs["display-rgb"],
        n_cols=5,
    )
    wandb.log({"sample_grid": wandb.Image(grid)})
    # wandb.log({"_sample_probabilities": wandb.Table(dataframe=results)})

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
