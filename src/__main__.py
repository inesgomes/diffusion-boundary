"""This is the main file for the diffusion-boundary package."""

import argparse
import math
import os
from datetime import datetime

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
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer

from src.classifier.factory import ClassifierFactory
from src.classifier.metrics import (
    BINARY_METRICS,
    MULTICLASS_METRICS,
    UNCERTAINTY_METRICS,
)
from src.dataset.aux import get_tst_dataset
from src.dataset.factory import DatasetFactory
from src.evaluation import (
    EVAL_METRICS,
    calculate_evaluation_metrics,
    calculate_feature_metrics,
    calculate_fid_metric,
    prepare_dataset_results,
)
from src.utils import generate_run_id, load_configurations
from src.visualization import (
    occlusion_map,
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


def save_results_to_disk(results: pd.DataFrame, name: str):
    """Save images to disk."""
    path = os.getenv("FILESDIR") + "/logs/" + wandb.run.id
    os.makedirs(path, exist_ok=True)
    results.to_pickle(f"{path}/results_{name}.pkl")
    print("Results saved at", path)


def get_valid_metrics(n_classes, dataset_columns):
    """Get which metrics are valid, given that it is not always possible to compute all metrics. Takes into considering if we have a binary or multiclass problem."""
    # all possible metrics
    sample_metrics = BINARY_METRICS if n_classes == 2 else MULTICLASS_METRICS
    eval_metrics_name = list(set(UNCERTAINTY_METRICS) | set(sample_metrics) | set(EVAL_METRICS))
    # select only valid metrics (withou NaNs)
    return [col for col in eval_metrics_name if col in dataset_columns]


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
        # k-LMS scheduler
        pipe.scheduler = LMSDiscreteScheduler.from_config(
            pipe.scheduler.config,
            rescale_betas_zero_snr=True,  # create images less noisy but nore blurry
            timestep_spacing="trailing",  # both together creates error
            prediction_type="epsilon",
            use_karras_sigmas=True,  # make sure we are using k-lms version
        )
        # from: https://huggingface.co/docs/diffusers/api/schedulers/ddim
        # DDIM scheduler
        # pipe.scheduler = DDIMScheduler.from_config(
        #    pipe.scheduler.config,
        #    rescale_betas_zero_snr=True,  # create images less noisy but nore blurry
        #    timestep_spacing="trailing",  # both together creates error
        #    prediction_type="epsilon",
        # )
        pipe.enable_attention_slicing()
        return pipe

    # Load and return the pipeline
    return pipeline_class.from_pretrained(
        model, custom_pipeline=custom_pipeline, cache_dir=os.getenv("HF_MODELS_CACHE")
    ).to(device)


def _build_prompt_from_strategy(diffusion_classes, strategy):
    """Return prompt according to the strategy and classes provided.

    Strategies:
    - "and"  -> 'class1 and class2 and ...'
    - "one<n>" -> 'class n'
    - "single" -> 'class1'; 'class2'; ...
    - anything else -> ''
    """
    # if strategy is "and", join all classes with " and "
    if strategy == "and":
        return [" and ".join(diffusion_classes)]
    # if strategy starts with "one" and has a number after
    if strategy.startswith("one") and len(strategy) > 3 and strategy[3] == "<" and strategy[-1] == ">":
        n = int(strategy[4:-1])
        if 1 <= n <= len(diffusion_classes):
            return [diffusion_classes[n - 1]]
    if strategy == "single":
        return diffusion_classes
    return [""]


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
        prompt = _build_prompt_from_strategy(diffusion_arguments["classes"], diffusion_arguments["prompt-strategy"])
        negative_prompt = (
            "" if diffusion_arguments["negative-prompt"] is None else diffusion_arguments["negative-prompt"]
        )
        print(">> Prompt: ", prompt)
        args.update(
            {
                "prompt": prompt,
                "labels_idx": classes_idx,
                "guidance_scale": diffusion_arguments["guidance-scale"],
                "guidance_rescale": diffusion_arguments["guidance-rescale"],
                "negative_prompt": negative_prompt,
            }
        )
    return args


def get_attribution_map(attr_type, classifier, dataset, idx_img, classes, device):
    """Get attribution map for a given classifier and an image from the dataset."""
    # get the image to be explained
    image, _ = dataset[idx_img]

    # dataframe with name, index and probability
    probs, _ = classifier.predict(image.unsqueeze(0).to(device))
    probs = probs.detach().cpu().numpy()[0]
    data = [
        (
            dataset.get_class_idx(class_name),
            dataset.class_labels[dataset.get_class_idx(class_name)],
            probs[dataset.get_class_idx(class_name)],
        )
        for class_name in classes
    ]
    target_info = pd.DataFrame(data, columns=["idx", "label", "prob"])

    if attr_type == "occlusion":
        return occlusion_map(classifier.get_model(), image, device, target_info)

    raise ValueError(f"Attribution map type {attr_type} not recognized.")


def generate_images(diffusion_settings, classifier, dataset, num_images, batch_size, seed, device):
    """Generate images using the diffusion pipeline described in the config file."""
    # get diffusion pipeline
    pipe = create_pipeline(
        diffusion_settings["type"], diffusion_settings["name"], diffusion_settings["pipeline"], device
    )
    # get arguments for the pipeline
    args = create_arguments(diffusion_settings["pipeline"], classifier, dataset, diffusion_settings["args"])

    # prepare prompts for all images
    if len(args["prompt"]) == 1:
        prompts_for_images = [args["prompt"][0]] * num_images
    else:
        # Evenly distribute prompts
        num_prompts = len(args["prompt"])
        prompts_for_images = [args["prompt"][i % num_prompts] for i in range(num_images)]

    # generate images in batches
    num_batches = math.ceil(num_images / batch_size)
    images = []
    generator = torch.Generator().manual_seed(seed)

    # generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(4)] # to generate batches
    for i in tqdm(range(num_batches), desc="Generating images"):
        # batch_size_to_use = min(batch_size, num_images - len(images))

        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_images)
        batch_prompts = prompts_for_images[start_idx:end_idx]

        args["prompt"] = batch_prompts

        log_denoising_images = i == 0

        batch_images = pipe(
            generator=generator,
            num_inference_steps=diffusion_settings["args"]["num-inference-steps"],
            batch_size=len(batch_prompts),
            log_denoising_images=log_denoising_images,
            **args,
        ).images
        images.extend(batch_images)

    print(f"Generated {len(images)} images")
    return images


def stress_test_evaluator(real_dataset, synth_dataset, classifier, boundary_classes, args):
    """Evaluate both real and synthetic datasets using metrics, and save results to disk if needed."""
    # compute reference dataset metrics that need the classifer
    real_dataset_res = prepare_dataset_results(
        real_dataset,
        classifier,
        boundary_classes,
        args["batch-size"],
        args["device"],
        args["mc-dropout"]["n-samples"],
        args["mc-dropout"]["threshold"],
    )

    # compute synthetic dataset metrics that need the classifer
    synth_dataset_res = prepare_dataset_results(
        synth_dataset,
        classifier,
        boundary_classes,
        args["batch-size"],
        args["device"],
        args["mc-dropout"]["n-samples"],
        args["mc-dropout"]["threshold"],
    )

    # compute synthetic metrics that need features and target (currently is only KDN)
    synth_metrics, real_features, fake_features = calculate_feature_metrics(
        real_dataset,
        synth_dataset,
        boundary_classes,
        args["batch-size"],
        args["device"],
    )
    synth_dataset_res = pd.concat([synth_dataset_res, synth_metrics], axis=1)
    valid_metrics = get_valid_metrics(args["n_classes"], synth_dataset_res.columns)

    # compute evaluation of synthetic dataset
    eval_metrics = calculate_evaluation_metrics(real_features, fake_features, synth_dataset_res, valid_metrics)

    # (fid needs to be calulated in a different way)
    eval_metrics["fid"] = calculate_fid_metric(real_dataset, synth_dataset, args["batch-size"], args["device"])

    if args["save-metrics-disk"]:
        save_results_to_disk(synth_dataset_res, "synthetic")
        save_results_to_disk(real_dataset_res, "real")

    return eval_metrics, valid_metrics, real_dataset_res, synth_dataset_res, real_features, fake_features


def stress_test_visualizations(
    real_dataset_res, real_features, real_labels, synth_dataset, synth_dataset_res, fake_features, valid_metrics, args
):
    """_summary_ Visualizations for the stress test.

    Args:
        real_features (_type_): _description_
        real_labels (_type_): _description_
        synth_dataset (_type_): _description_
        synth_dataset_res (_type_): _description_
        diffusion_config (_type_): _description_
        args (_type_): _description_
    """
    # umap visualization of features
    features_umap = visualize_features_umap(
        real_features, real_labels, fake_features, synth_dataset_res[args["args"]["guidance"]]
    )

    # top n images with guidance metric
    fig = visualize_top_synthetic_metric(
        synth_dataset,
        synth_dataset_res,
        sort_metric=args["args"]["guidance"],
        ascending=True,  # minimize
        top_n=5,
        display_rgb=args["display-rgb"],
    )

    # melt real and synthetic values
    real_vs_synth = (
        pd.concat([real_dataset_res, synth_dataset_res], keys=["Real", "Synthetic"])
        .reset_index()
        .rename(columns={"level_0": "keys"})
        .drop(columns=["level_1"])
    )

    # metric distribution (real vs fake) - boxplot
    dist_metric = visualize_metrics_distributions(real_vs_synth, valid_metrics)

    # class distributions for top classes - kde plot
    dist_labels = visualize_class_distributions(
        real_vs_synth, classes=args["args"]["classes"], n_classes=args["n_classes"]
    )

    # ambiguity matrix (only if low number of classes)
    viz_pairs = None
    table_confusion = None
    if args["n_classes"] <= 10:
        viz_pairs, table_confusion = visualize_confusion(
            real_dataset_res,
            synth_dataset_res,
            args["n_classes"],
            args["certainty-threshold"],
        )

    return features_umap, fig, dist_metric, dist_labels, viz_pairs, table_confusion


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
            "classsifier": (
                classifier_config["name"] + "_corrupt"
                if classifier_config["corrupt"] > 0
                else classifier_config["name"]
            ),
            "diffusion": diffusion_config_txt,
            "save-metrics-disk": default_configs["save-metrics-disk"],
            "save-images-disk": default_configs["save-images-disk"],
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
        if classifier_config["corrupt"] > 0:
            classifier.soft_corrupt_classifier(classifier_config["corrupt"])
        if classifier_config["calibrate"]:
            # prepare calibration dataset
            calib_images, calib_labels, calib_class_labels = get_tst_dataset(
                dataset_config["name"],
                "validation",
                300,  # I just need 300 images for calibration
                dataset_config["subset"],
            )
            calib_dataset = DatasetFactory.dataset_from_lib(
                classifier_config["lib"],
                classifier_config["name"],
                dataset_config["name"],
                dataset_config["n_classes"],
                calib_class_labels,
                calib_images,
            )
            calib_dataset.set_labels(calib_labels)
            calibloader = DataLoader(calib_dataset, batch_size=10, shuffle=False, num_workers=6)
            # calibrate classifier
            classifier.calibrate(calibloader)

    # get original dataset, to find the labels
    real_images, real_labels, class_labels = get_tst_dataset(
        dataset_config["name"],
        dataset_config["split"],
        max(evaluation_config["num-images"], dataset_config["num-images"]),
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

    # uncomment if we want to get the images from disk
    # images_path = os.getenv("FILESDIR") + "/logs/" + run_id + "/images.pkl"
    # with open(images_path, "rb") as f:
    #    images = torch.load(f)

    # generate images
    images = generate_images(
        diffusion_config,
        classifier,
        synth_dataset,
        evaluation_config["num-images"],
        default_configs["batch-size"],
        default_configs["seed"],
        default_configs["device"],
    )
    synth_dataset.set_images(images)

    torch.cuda.empty_cache()

    # if we need to generate only 1 image, finish without evaluation
    if evaluation_config["num-images"] == 1:
        # TODO: consider applying UMAP to see where the image lies in the feature space - use all classes that appear in the top 4 over selected image
        wandb.finish()
        return 0

    # save images to disk, if needed
    if default_configs["save-images-disk"]:
        save_images_to_disk(images)

    # evaluate both datasets
    args = {**default_configs, **evaluation_config, **dataset_config}
    eval_metrics, valid_metrics, real_dataset_res, synth_dataset_res, real_features, fake_features = (
        stress_test_evaluator(real_dataset, synth_dataset, classifier, diffusion_config["args"]["classes"], args)
    )

    # log metrics
    wandb.log(eval_metrics)

    # visualizations

    # sample: grid of images and respective probs
    print("Visualizing sample images...")
    grid, _ = visualize_sample_synthetic_images(
        synth_dataset,
        synth_dataset_res,
        evaluation_config["viz-sample-size"],
        diffusion_config["args"]["guidance"],  # if diffusion_config["pipeline"] == "guidance" else "entropy",
        default_configs["display-rgb"],
        n_cols=5,
        sort=True,  # True if we want to see best samples
    )
    wandb.log({"sample_grid": wandb.Image(grid)})
    # wandb.log({"_sample_probabilities": wandb.Table(dataframe=results)})

    if default_configs["log-plots"]:
        print("Generating visualizations...")
        args = {**args, **diffusion_config}
        features_umap, fig_sample, dist_metric, dist_labels, viz_pairs, table_confusion = stress_test_visualizations(
            real_dataset_res,
            real_features,
            real_labels,
            synth_dataset,
            synth_dataset_res,
            fake_features,
            valid_metrics,
            args,
        )

        wandb.log({"umap": wandb.Image(features_umap)})
        wandb.log({f"{diffusion_config['args']['guidance']}_sample": wandb.Image(fig_sample)})
        wandb.log({"dist_metrics": wandb.Image(dist_metric)})
        wandb.log({"dist_labels": wandb.Image(dist_labels)})
        if viz_pairs is not None and table_confusion is not None:
            wandb.log({"pairs_cm": wandb.Image(viz_pairs)})
            wandb.log({"_boundaries": wandb.Table(dataframe=table_confusion)})

    if evaluation_config["attr-map"]:
        print("Generating attribution map...")
        attr_map = get_attribution_map(
            "occlusion",
            classifier,
            synth_dataset,
            0,
            diffusion_config["args"]["classes"],
            default_configs["device"],
        )
        wandb.log({"occlusion_map": wandb.Image(attr_map)})

    # finish wandb
    wandb.finish()
    return 0


def main(configuration):
    """Run the stress test per configuration."""
    user_configs = configuration["user-args"]
    dataset_config = configuration["dataset"]
    classifier_config = configuration["classifier"]
    evaluation_config = configuration["evaluation"]

    guidance_metric = configuration["diffusion"]["args"]["guidance"]
    alpha = configuration["diffusion"]["args"]["alpha"]
    guidance_scale = configuration["diffusion"]["args"]["guidance-scale"]
    diffusion_config = configuration["diffusion"]

    # the group will be a timestamp that this main started, so that we can join multiple runs
    group_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    i = 1
    max_i = len(guidance_metric) * len(alpha) * len(guidance_scale)
    for guidance_metric_value in guidance_metric:
        for alpha_value in alpha:
            for guidance_scale_value in guidance_scale:
                diffusion_config["args"]["guidance"] = guidance_metric_value
                diffusion_config["args"]["alpha"] = alpha_value
                diffusion_config["args"]["guidance-scale"] = guidance_scale_value

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
