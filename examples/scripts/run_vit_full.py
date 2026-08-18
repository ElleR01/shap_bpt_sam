#!/usr/bin/env python
"""Run a ViT-B/16 + ShapBPT full-image-set experiment.

This is the classifier counterpart of run_yolo_full.py. It reuses the same
partition, AUC, timing, and HTML summary helpers, but replaces YOLO detection
with ViT-B/16 ImageNet classification.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from math import ceil
from pathlib import Path

PROJECT_ROOT_HINT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_HINT))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap_bpt
import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision.models import ViT_B_16_Weights, vit_b_16
from tqdm.auto import tqdm

from run_yolo_full import (
    METHOD_ORDER,
    PARTITION_TIME_BY_METHOD,
    BACKGROUND_REPLACEMENT_DETAILS,
    add_exp_suffix,
    aggregate_auc_outputs,
    build_partition_summary,
    build_xai_results_webpage,
    collect_image_ids,
    describe_compute_resources,
    find_project_root,
    format_duration,
    image_has_segmentation,
    infer_experiment_no,
    iteration_resource_summary,
    load_config,
    load_image,
    normalize_image_id,
    reset_iteration_resource_counters,
    save_current_auc_table,
    save_run_info,
    save_time_comparison_outputs,
    save_total_time_outputs,
    select_device,
)


def load_vit_b_16_model(args, device: torch.device):
    weights = None if args.weights == "none" else getattr(ViT_B_16_Weights, args.weights)
    model = vit_b_16(weights=weights)
    categories = weights.meta["categories"] if weights is not None else [str(i) for i in range(1000)]

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
        cleaned = {key.replace("module.", ""): value for key, value in state_dict.items()}
        model.load_state_dict(cleaned)

    model.to(device)
    model.eval()
    preprocess = weights.transforms() if weights is not None else ViT_B_16_Weights.IMAGENET1K_V1.transforms()
    return model, preprocess, categories


def image_to_uint8(image) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def preprocess_vit_batch(images, preprocess, device: torch.device):
    tensors = []
    for image in images:
        pil_image = Image.fromarray(image_to_uint8(image))
        tensors.append(preprocess(pil_image))
    return torch.stack(tensors, dim=0).to(device, non_blocking=True)


def predict_vit_batch(model, preprocess, device: torch.device, images, batch_size: int = 128):
    images = list(images)
    if not images:
        return np.empty((0, 1000))

    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            batch = preprocess_vit_batch(batch_images, preprocess, device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            outputs.append(probs.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def make_predict_vit_masked(model, preprocess, device, image_to_explain, background_image_set, batch_size: int):
    def predict_vit_masked(masks, verbose: bool = False):
        masked_images = []
        mask_indexes = []
        for mask_index, mask in enumerate(masks):
            if len(mask.shape) == 2:
                mask3 = np.stack([mask, mask, mask], axis=2)
            else:
                mask3 = mask.copy()
            if mask3.shape[:2] != image_to_explain.shape[:2]:
                raise ValueError(f"Mask shape {mask3.shape[:2]} does not match image shape {image_to_explain.shape[:2]}")
            mask3 = mask3.astype(bool)
            for repl in background_image_set:
                masked_images.append(np.where(mask3, image_to_explain, repl))
                mask_indexes.append(mask_index)

        batch_preds = predict_vit_batch(model, preprocess, device, masked_images, batch_size=batch_size)
        preds = np.zeros((len(masks), batch_preds.shape[1]))
        counts = np.zeros(len(masks))
        for mask_index, pred in zip(mask_indexes, batch_preds):
            preds[mask_index] += pred
            counts[mask_index] += 1
        return preds / counts[:, None]

    return predict_vit_masked


def load_vit_image_to_explain(image_id: str, image_dir: str, model, preprocess, categories, device, bg_type: str, batch_size: int):
    image_path = Path(image_dir) / f"{image_id}.jpg"
    image, image_tensor, background_image_set, background_tensors = load_image(image_path, device=device, bg_type=bg_type)
    predicted_fS = predict_vit_batch(model, preprocess, device, [image], batch_size=batch_size)[0]
    predicted_f0 = predict_vit_batch(model, preprocess, device, background_image_set, batch_size=batch_size).mean(axis=0)
    sorted_classes = np.flip(np.argsort(predicted_fS))
    predicted_cls = int(sorted_classes[0])
    return {
        "fname": image_id,
        "image_to_explain": image,
        "image_to_explain_tensor": image_tensor,
        "background_image_set": background_image_set,
        "background_tensors": background_tensors,
        "predicted_fS": predicted_fS,
        "sorted_classes": sorted_classes,
        "sorted_probs": predicted_fS[sorted_classes],
        "predicted_cls": predicted_cls,
        "fixed_category": categories[predicted_cls],
        "f_S": float(predicted_fS[predicted_cls]),
        "predicted_f0": predicted_f0,
        "f_0": float(predicted_f0[predicted_cls]),
    }


def top_k_classes(predictions: np.ndarray, categories, k: int = 5):
    indexes = np.flip(np.argsort(predictions))[:k]
    return [(int(idx), str(categories[int(idx)]), float(predictions[int(idx)])) for idx in indexes]


def save_vit_predictions(input_data, path_results: Path, has_segmentation: bool, top_classes, model_name: str):
    image_no = int(input_data["fname"])
    path_results_img = path_results / str(image_no)
    path_results_img.mkdir(parents=True, exist_ok=True)
    summary = {
        "image_id": str(input_data["fname"]),
        "model_type": "vit_b_16",
        "model_name": model_name,
        "fixed_category": input_data["fixed_category"],
        "explained_class": input_data["explained_class"],
        "f_S": float(input_data["f_S"]),
        "f_0": float(input_data["f_0"]),
        "has_segmentation": bool(has_segmentation),
        "speed": {},
        "top_k_classes": [
            {"class_id": class_id, "class_name": class_name, "confidence": confidence}
            for class_id, class_name, confidence in top_classes
        ],
    }
    out_path = path_results_img / f"{image_no}_predictions.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_vit_xai(input_data, partitions, shap_values, categories, image_id, save_path: Path, save_fig=False, destroy_fig=False):
    import utils_sam as uts
    import utils_xai as utx

    n_panels = 3 + len(shap_values)
    ncols = 3
    nrows = ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.6 * nrows))
    axes = np.atleast_1d(axes).flatten()

    axes[0].imshow(input_data["image_to_explain"])
    axes[0].set_title(f"Original Image - {image_id}")
    axes[0].axis("off")

    top_lines = []
    for rank, cls in enumerate(input_data["sorted_classes"][:5], start=1):
        top_lines.append(f"{rank}. {categories[int(cls)]}: {input_data['predicted_fS'][int(cls)]:.4f}")
    axes[1].axis("off")
    axes[1].set_title("ViT-B/16 Top Classes")
    axes[1].text(0.0, 0.95, "\n".join(top_lines), va="top", ha="left", fontsize=11, transform=axes[1].transAxes)

    mask_type = "sam" if "sam" in partitions else next(iter(partitions), None)
    if mask_type is not None:
        masks = partitions[mask_type]
        n_labels = int(np.max(masks))
        cmap, norm = uts.black_rainbow_colormap(n_labels, rainbow_name="turbo")
        image = axes[2].imshow(masks, cmap=cmap, norm=norm, aspect="auto")
        fig.colorbar(image, ax=axes[2], fraction=0.03)
        axes[2].set_title(f"{mask_type}: {len(np.unique(masks))} labels")
        axes[2].set_xticks([])
        axes[2].set_yticks([])
    else:
        axes[2].axis("off")
        axes[2].set_title("No partitions")

    shapley_values_colormap = utx.get_shapley_values_colormap()
    for panel_index, (method, values) in enumerate(shap_values.items(), start=3):
        scaled, max_value = utx.scale_shap_values(values, robust_percentile=99.5, factor=1)
        im = axes[panel_index].imshow(scaled[0], vmin=-max_value, vmax=max_value, cmap=shapley_values_colormap)
        fig.colorbar(im, ax=axes[panel_index], fraction=0.03)
        axes[panel_index].set_title(f"{method} Explanation")
        axes[panel_index].set_xticks([])
        axes[panel_index].set_yticks([])

    for ax in axes[n_panels:]:
        ax.axis("off")

    plt.tight_layout()
    if save_fig:
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f"{image_id}_xai_results.png", dpi=100, bbox_inches="tight")
    if destroy_fig:
        plt.close(fig)
    else:
        plt.show()
    return fig


def run(args):
    project_root = find_project_root()
    scripts_dir = project_root / "examples" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import utils_sam_2 as uts2
    import utils_xai as utx

    config = load_config(project_root, args.config)
    if args.output_dir:
        config.setdefault("output", {})
        config["output"]["dir"] = str(Path(args.output_dir).expanduser())
        config["output"]["folder"] = ""

    device = select_device(args.device or config.get("device", "auto"))
    model, preprocess, categories = load_vit_b_16_model(args, device)

    print(f"Project root: {project_root}")
    print(f"Device: {device}")
    print(f"Model: ViT-B/16")
    print(f"ViT weights: {args.weights}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    print(f"Classes: {len(categories)}")
    print(f"shap_bpt version: {shap_bpt.__version__}")

    resource_logging = args.verbose_level in {"medium", "high"}
    if resource_logging:
        print(
            "Run settings: "
            f"max_evals={args.max_evals}, "
            f"explain_batch_size={args.eval_batch_size}, "
            f"auc_eval_batch_size={args.auc_batch_size}, "
            f"vit_batch_size={args.vit_batch_size}, "
            f"methods={','.join(METHOD_ORDER)}, "
            f"num_explained_classes={args.num_explained_classes}, "
            f"bg_type={args.bg_type}, "
            f"verbose_k={args.verbose_k}"
        )
        describe_compute_resources(device)

    masks_base_path = Path(config["data"]["masks_path"]) / config["data"]["mask_dir"] / config["data"]["mask_dir_final"]
    image_dir = config["data"]["image_dir"]
    exp_no = infer_experiment_no("vit_b_16", args.bg_type, args.exp_no)
    results_name = add_exp_suffix(args.results_name, exp_no if args.exp_suffix else None)
    path_results = Path(config["output"]["dir"]) / config["output"].get("folder", "") / results_name
    path_results.mkdir(parents=True, exist_ok=True)
    save_run_info(
        path_results,
        {
            "exp_no": exp_no,
            "model_group": "ViT-B/16",
            "model": "vit_b_16",
            "weights": args.weights,
            "checkpoint": args.checkpoint,
            "num_model_classes": len(categories),
            "background_replacement_values": args.bg_type,
            "background_details": BACKGROUND_REPLACEMENT_DETAILS.get(args.bg_type, "-"),
            "class_aggregation": None,
            "max_evals": args.max_evals,
            "eval_batch_size": args.eval_batch_size,
            "auc_batch_size": args.auc_batch_size,
            "vit_batch_size": args.vit_batch_size,
            "num_explained_classes": args.num_explained_classes,
            "methods": METHOD_ORDER,
            "config": args.config,
            "masks_base_path": str(masks_base_path),
            "image_dir": str(image_dir),
            "results_dir": str(path_results),
        },
    )

    coco = None
    annotation_file = config["data"].get("annotation_file")
    if annotation_file:
        annotation_path = Path(annotation_file)
        if not annotation_path.is_absolute():
            annotation_path = project_root / annotation_path
        if annotation_path.exists():
            coco = COCO(str(annotation_path))

    image_ids = [normalize_image_id(x) for x in args.image_ids] if args.image_ids else collect_image_ids(masks_base_path)
    if args.limit is not None:
        image_ids = image_ids[: args.limit]

    print(f"Images to process: {len(image_ids)}")
    print(f"Masks: {masks_base_path}")
    print(f"Results: {path_results}")

    partition_summary_df = build_partition_summary(image_ids, masks_base_path, path_results, uts2)

    failures = []
    auc_table_rows = []
    iteration_durations = []
    method_progress = args.verbose_level in {"medium", "high"}

    for iteration_index, image_id in enumerate(tqdm(image_ids, desc="Full ViT XAI", position=0, dynamic_ncols=True), start=1):
        iteration_start = time.time()
        first_iteration_breakdown = {
            "load_image_and_target": 0.0,
            "segmentation_lookup": 0.0,
            "save_predictions": 0.0,
            "load_partition_json": 0.0,
            "load_partitions": 0.0,
            "build_bpt": 0.0,
            "explanation": 0.0,
            "single_attribution_plots": 0.0,
            "xai_grid_plot": 0.0,
            "auc_compute": 0.0,
            "auc_plot": 0.0,
        }
        first_iteration_method_times = {}
        if resource_logging:
            reset_iteration_resource_counters(device)

        image_no = int(image_id)
        try:
            if args.verbose:
                print("=" * 100)
                print(f"Image: {image_id}")

            timing_start = time.time()
            input_data = load_vit_image_to_explain(
                image_id,
                image_dir,
                model=model,
                preprocess=preprocess,
                categories=categories,
                device=device,
                bg_type=args.bg_type,
                batch_size=args.vit_batch_size,
            )
            first_iteration_breakdown["load_image_and_target"] += time.time() - timing_start

            fixed_category = input_data["fixed_category"]
            timing_start = time.time()
            has_segmentation = image_has_segmentation(coco, image_no) if coco is not None else False
            first_iteration_breakdown["segmentation_lookup"] += time.time() - timing_start

            path_results_img = path_results / str(image_no)
            path_results_img.mkdir(parents=True, exist_ok=True)

            predict_masked = make_predict_vit_masked(
                model,
                preprocess,
                device,
                input_data["image_to_explain"],
                input_data["background_image_set"],
                batch_size=args.vit_batch_size,
            )
            explainer = shap_bpt.Explainer(
                predict_masked,
                input_data["image_to_explain"],
                num_explained_classes=args.num_explained_classes,
                verbose=args.verbose_level == "high",
            )
            predicted_cls = int(explainer.output_indexes[0])
            f_S = float(explainer.base_nuN[0])
            f_0 = float(explainer.base_nu0[0])
            input_data["explained_class"] = categories[predicted_cls]
            input_data["f_S"] = f_S
            input_data["f_0"] = f_0

            timing_start = time.time()
            top_classes = top_k_classes(input_data["predicted_fS"], categories, k=args.top_k)
            save_vit_predictions(input_data, path_results, has_segmentation, top_classes, model_name="vit_b_16")
            first_iteration_breakdown["save_predictions"] += time.time() - timing_start

            partitions, partition_capped, shap_values, auc_records = {}, {}, {}, []
            timing_start = time.time()
            partition_results = uts2.load_json_partitions(str(masks_base_path), image_id, verbose=False)
            first_iteration_breakdown["load_partition_json"] += time.time() - timing_start

            methods_iter = tqdm(
                METHOD_ORDER,
                desc=f"{image_id} methods",
                leave=False,
                position=1,
                dynamic_ncols=True,
                disable=not method_progress,
            )
            for method in methods_iter:
                if method_progress:
                    methods_iter.set_postfix_str(method)
                if method == "BPT":
                    bptree = None
                else:
                    timing_start = time.time()
                    partitions[method], partition_capped[method] = uts2.load_refine_partitions(
                        input_data["image_to_explain"], str(masks_base_path), image_id, partition_type=method
                    )
                    first_iteration_breakdown["load_partitions"] += time.time() - timing_start
                    timing_start = time.time()
                    bptree = shap_bpt.build_bpt_from_image(input_data["image_to_explain"], prebuilt_partitions=partition_capped[method])
                    first_iteration_breakdown["build_bpt"] += time.time() - timing_start

                start = time.time()
                shap_values[method] = explainer.explain_instance(
                    args.max_evals,
                    bpt=bptree,
                    method="BPT",
                    batch_size=args.eval_batch_size,
                )
                time_exp = time.time() - start
                first_iteration_breakdown["explanation"] += time_exp
                first_iteration_method_times[method] = time_exp
                if method_progress:
                    methods_iter.set_postfix_str(f"{method}, exp={format_duration(time_exp)}")
                if args.verbose_level == "high":
                    print(f"{method}: Shapley sum={np.sum(shap_values[method][0])}")

                if args.save_plots:
                    timing_start = time.time()
                    utx.plot_single_attributions(
                        shap_values[method],
                        str(path_results_img),
                        f"{method}_{image_no}.png",
                        robust_percentile=99.9985,
                        save_plot=True,
                        destroy_fig=True,
                    )
                    first_iteration_breakdown["single_attribution_plots"] += time.time() - timing_start

                partition_time_col = PARTITION_TIME_BY_METHOD.get(method)
                auc_records.append(
                    {
                        "label": method,
                        "shap_values": shap_values[method][0],
                        "time_partition": partition_results.get(partition_time_col) if partition_time_col else None,
                        "time_exp": time_exp,
                    }
                )

            if args.save_plots:
                timing_start = time.time()
                plot_vit_xai(
                    input_data,
                    partitions,
                    shap_values,
                    categories,
                    image_id,
                    save_path=path_results_img,
                    save_fig=True,
                    destroy_fig=True,
                )
                first_iteration_breakdown["xai_grid_plot"] += time.time() - timing_start

            if args.compute_auc:
                timing_start = time.time()
                auc_results = utx.compute_auc_results(
                    auc_records,
                    predict_masked,
                    f_S,
                    f_0,
                    predicted_cls,
                    batch_size=args.auc_batch_size,
                    verbose=args.verbose_level == "high",
                )
                first_iteration_breakdown["auc_compute"] += time.time() - timing_start
                if args.save_plots:
                    timing_start = time.time()
                    utx.plot_auc_results(
                        auc_results,
                        str(path_results_img),
                        save_plot=True,
                        image_no=image_no,
                        destroy_figs=True,
                    )
                    first_iteration_breakdown["auc_plot"] += time.time() - timing_start
                auc_table_rows.extend(
                    utx.auc_results_to_rows(
                        auc_results,
                        image_no=image_no,
                        image_id=image_id,
                        fixed_category=fixed_category,
                        f_S=f_S,
                        f_0=f_0,
                    )
                )
                current_run_path, current_auc_df = save_current_auc_table(auc_table_rows, path_results)
                if resource_logging and (iteration_index == 1 or (args.verbose_k > 0 and iteration_index % args.verbose_k == 0)):
                    n_saved_images = current_auc_df["image_no"].nunique() if "image_no" in current_auc_df.columns else "unknown"
                    tqdm.write(f"Current-run AUC saved: {current_run_path} ({n_saved_images} images)")

        except Exception as exc:
            failure = {"image_id": image_id, "error": repr(exc), "traceback": traceback.format_exc()}
            failures.append(failure)
            print(f"FAILED {image_id}: {exc}")
            if args.fail_fast:
                raise
        finally:
            iteration_duration = time.time() - iteration_start
            iteration_durations.append(iteration_duration)
            should_print_resources = (
                resource_logging
                and (iteration_index == 1 or (args.verbose_k > 0 and iteration_index % args.verbose_k == 0))
            )
            if should_print_resources:
                pending = len(image_ids) - iteration_index
                avg_time = sum(iteration_durations) / len(iteration_durations)
                eta = avg_time * pending
                resource_text = iteration_resource_summary(device)
                tqdm.write(
                    f"Resources after {image_id} [{iteration_index}/{len(image_ids)}]: "
                    f"iteration={format_duration(iteration_duration)}, "
                    f"avg={format_duration(avg_time)}, "
                    f"pending={pending}, "
                    f"eta={format_duration(eta)}"
                    f"{resource_text}"
                )
            if resource_logging and iteration_index == 1:
                measured = sum(first_iteration_breakdown.values())
                other = max(0.0, iteration_duration - measured)
                breakdown_parts = [
                    f"total={format_duration(iteration_duration)}",
                    f"load={format_duration(first_iteration_breakdown['load_image_and_target'])}",
                    f"seg={format_duration(first_iteration_breakdown['segmentation_lookup'])}",
                    f"save_pred={format_duration(first_iteration_breakdown['save_predictions'])}",
                    f"partition_json={format_duration(first_iteration_breakdown['load_partition_json'])}",
                    f"partition_npys={format_duration(first_iteration_breakdown['load_partitions'])}",
                    f"build_bpt={format_duration(first_iteration_breakdown['build_bpt'])}",
                    f"explain={format_duration(first_iteration_breakdown['explanation'])}",
                    f"single_plots={format_duration(first_iteration_breakdown['single_attribution_plots'])}",
                    f"xai_plot={format_duration(first_iteration_breakdown['xai_grid_plot'])}",
                    f"auc_compute={format_duration(first_iteration_breakdown['auc_compute'])}",
                    f"auc_plot={format_duration(first_iteration_breakdown['auc_plot'])}",
                    f"other={format_duration(other)}",
                ]
                tqdm.write("First image timing breakdown: " + ", ".join(breakdown_parts))
                if first_iteration_method_times:
                    method_parts = [f"{method}={format_duration(first_iteration_method_times[method])}" for method in METHOD_ORDER if method in first_iteration_method_times]
                    tqdm.write("First image explanation by method: " + ", ".join(method_parts))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if auc_table_rows:
        current_run_path, auc_all_df = save_current_auc_table(auc_table_rows, path_results)
        print(f"Current-run AUC table: {current_run_path}")
        aggregate_auc_outputs(auc_all_df, partition_summary_df, path_results, args.max_evals)
        save_time_comparison_outputs(auc_all_df, partition_summary_df, path_results, args.max_evals)
        save_total_time_outputs(auc_all_df, partition_summary_df, path_results, args.max_evals)

    if failures:
        failures_path = path_results / "full_image_set_failures.json"
        failures_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"Failures saved to: {failures_path}")

    if args.build_html:
        build_xai_results_webpage(path_results)

    print(f"Completed images: {len(image_ids) - len(failures)}")
    print(f"Failures: {len(failures)}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="MSCOCO_mac", help="Config name from examples/configs or path to YAML.")
    parser.add_argument("--weights", default="IMAGENET1K_V1", choices=["IMAGENET1K_V1", "DEFAULT", "none"])
    parser.add_argument("--checkpoint", default=None, help="Optional ViT-B/16 checkpoint path.")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"], help="Override config device.")
    parser.add_argument("--output-dir", default=None, help="Override config output.dir.")
    parser.add_argument("--results-name", default="xai_results", help="Base results folder name before experiment suffix.")
    parser.add_argument("--exp-no", default=None, help="Override experiment number, e.g. E3_1.")
    parser.add_argument("--no-exp-suffix", dest="exp_suffix", action="store_false", help="Do not append experiment number to results folder.")
    parser.add_argument("--max-evals", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images for a smoke test.")
    parser.add_argument("--image-ids", nargs="*", default=None, help="Optional explicit image ids.")
    parser.add_argument("--bg-type", default="noise", choices=["black", "gray", "white", "blurred", "noise", "full"])
    parser.add_argument("--num-explained-classes", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--auc-batch-size", type=int, default=128)
    parser.add_argument("--vit-batch-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--verbose-level", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--verbose-k", type=int, default=10, help="Print medium/high resource summary every K images after image 1.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-auc", dest="compute_auc", action="store_false")
    parser.add_argument("--no-plots", dest="save_plots", action="store_false")
    parser.add_argument("--no-html", dest="build_html", action="store_false")
    parser.set_defaults(compute_auc=True, save_plots=True, build_html=True, exp_suffix=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
