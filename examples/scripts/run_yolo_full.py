#!/usr/bin/env python
"""Run the YOLO + ShapBPT full-image-set experiment.

This script is a cleaned, runnable version of
examples/notebooks/Object_yolo_ms_coco_full_image_set_tests.ipynb.
It computes explanations, AUC metrics, timing summaries, plots, and a
single HTML report from saved per-image results.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
import traceback
from pathlib import Path
from urllib.parse import unquote

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
import yaml
from pycocotools.coco import COCO
from scipy.ndimage import gaussian_filter
from skimage.filters import gaussian
from torchvision import transforms
from tqdm.auto import tqdm
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset, safe_download


METHOD_ORDER = ["BPT", "sam", "coverage", "compact", "filled", "refined", "refined_filled"]
PARTITION_TIME_BY_METHOD = {
    "BPT": None,
    "sam": "time_sam",
    "coverage": "time_coverage",
    "compact": "time_compact",
    "filled": "time_filler",
    "refined": "time_refined",
    "refined_filled": "time_refined_filled",
}


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start).resolve()
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists() and (path / "shap_bpt").is_dir():
            return path
    raise FileNotFoundError("Could not find the project root.")


def load_config(project_root: Path, config_name_or_path: str) -> dict:
    config_path = Path(config_name_or_path)
    if not config_path.suffix:
        config_path = project_root / "examples" / "configs" / f"{config_name_or_path}.yaml"
    if not config_path.is_absolute():
        config_path = project_root / config_path
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def resolve_checkpoint_dir(project_root: Path, checkpoint_dir: str | None = None) -> Path:
    path = Path(checkpoint_dir).expanduser() if checkpoint_dir else project_root / "examples" / "notebooks" / "checkpoints"
    if not path.is_absolute():
        path = project_root / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_search_dirs(project_root: Path, primary: Path) -> list[Path]:
    dirs = [
        primary,
        project_root / "examples" / "notebooks" / "checkpoints",
        project_root / "examples" / "checkpoints",
        project_root / "checkpoints",
    ]
    unique_dirs = []
    for path in dirs:
        path = path.resolve()
        if path not in unique_dirs:
            unique_dirs.append(path)
    return unique_dirs


def resolve_yolo_source(project_root: Path, model_name: str, weights: str | None = None, checkpoint_dir: str | None = None) -> str:
    checkpoint_path = resolve_checkpoint_dir(project_root, checkpoint_dir)
    search_dirs = checkpoint_search_dirs(project_root, checkpoint_path)
    if weights:
        weights_text = str(weights)
        if weights_text.lower().startswith(("https://", "http://")):
            target = checkpoint_path / Path(unquote(weights_text.split("?")[0])).name
            if not target.exists():
                safe_download(url=weights_text, file=target, unzip=False)
            return str(target)

        weights_path = Path(weights_text).expanduser()
        if not weights_path.is_absolute():
            weights_path = project_root / weights_path
        return str(weights_path)

    model_text = str(model_name)
    model_path = Path(model_text).expanduser()
    if model_path.suffix or "/" in model_text:
        if not model_path.is_absolute():
            model_path = project_root / model_path
        return str(model_path)

    candidates = []
    for stem in (model_text, model_text.replace("_", "")):
        for search_dir in search_dirs:
            candidates.append(search_dir / f"{stem}.pt")
            candidates.append(search_dir / stem)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    attempted = []
    for stem in (model_text, model_text.replace("_", "")):
        asset_name = stem if stem.endswith(".pt") else f"{stem}.pt"
        target = checkpoint_path / asset_name
        attempted.append(str(target))
        try:
            downloaded = Path(attempt_download_asset(target, release="latest"))
            if downloaded.exists():
                return str(downloaded)
        except Exception:
            pass

    raise FileNotFoundError(
        "Could not find or download YOLO checkpoint. "
        f"Looked in: {[str(path) for path in search_dirs]}. Attempted downloads: {attempted}. "
        f"For a custom model, place it at {checkpoint_path / (model_text + '.pt')} "
        "or pass --weights /path/to/model.pt or --weights https://.../model.pt."
    )


def select_device(name: str = "auto") -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def bytes_to_gib(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) / (1024 ** 3):.2f} GiB"


def cuda_device_index(device: torch.device) -> int:
    if device.type == "cuda" and device.index is not None:
        return device.index
    return torch.cuda.current_device()


def cuda_memory_snapshot(device: torch.device) -> dict:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    try:
        index = cuda_device_index(device)
        free_mem, total_mem = torch.cuda.mem_get_info(index)
        allocated = torch.cuda.memory_allocated(index)
        reserved = torch.cuda.memory_reserved(index)
        peak_allocated = torch.cuda.max_memory_allocated(index)
        peak_reserved = torch.cuda.max_memory_reserved(index)
        return {
            "free": free_mem,
            "total": total_mem,
            "allocated": allocated,
            "reserved": reserved,
            "peak_allocated": peak_allocated,
            "peak_reserved": peak_reserved,
            "peak_allocated_pct": (peak_allocated / total_mem * 100) if total_mem else None,
            "peak_reserved_pct": (peak_reserved / total_mem * 100) if total_mem else None,
        }
    except Exception:
        return {}


def describe_compute_resources(device: torch.device) -> None:
    print("Compute resources:")
    print(f"  Selected device: {device}")
    print(f"  CPU cores: {os.cpu_count() or 'unknown'}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")

    if torch.cuda.is_available():
        index = cuda_device_index(device) if device.type == "cuda" else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        snapshot = cuda_memory_snapshot(torch.device(f"cuda:{index}"))
        print(f"  CUDA devices: {torch.cuda.device_count()}")
        print(f"  GPU name: {torch.cuda.get_device_name(index)}")
        print(f"  GPU total memory: {bytes_to_gib(props.total_memory)}")
        if snapshot:
            print(f"  GPU free memory now: {bytes_to_gib(snapshot['free'])}")
            print(f"  GPU reserved by PyTorch now: {bytes_to_gib(snapshot['reserved'])}")
    elif device.type == "mps":
        print("  GPU name: Apple Metal Performance Shaders")
        print("  GPU memory detail: not exposed by PyTorch MPS like CUDA.")
    else:
        print("  GPU name: n/a")
        print("  GPU memory detail: n/a")


def reset_iteration_resource_counters(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.synchronize(cuda_device_index(device))
            torch.cuda.reset_peak_memory_stats(cuda_device_index(device))
        except Exception:
            pass


def iteration_resource_summary(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.synchronize(cuda_device_index(device))
        except Exception:
            pass
        snapshot = cuda_memory_snapshot(device)
        if snapshot:
            return (
                f", peak_gpu_mem={bytes_to_gib(snapshot['peak_allocated'])}"
                f" ({snapshot['peak_allocated_pct']:.1f}% allocated),"
                f" peak_reserved={bytes_to_gib(snapshot['peak_reserved'])}"
                f" ({snapshot['peak_reserved_pct']:.1f}% reserved)"
            )
        return ", peak_gpu_mem=n/a"

    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        try:
            return f", mps_allocated={bytes_to_gib(torch.mps.current_allocated_memory())}"
        except Exception:
            return ", mps_allocated=n/a"

    return ""


def normalize_image_id(value) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(12)


def load_image(fname: str | Path, device: torch.device, bg_type: str = "noise", im_size=None):
    img_bgr = cv2.imread(str(fname))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {fname}")
    image_to_explain = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if im_size is not None:
        image_to_explain = cv2.resize(image_to_explain, im_size)

    np.random.seed(0)
    bkgnd0 = np.full_like(image_to_explain, 0)
    bkgnd1 = np.full_like(image_to_explain, 127)
    bkgnd2 = np.full_like(image_to_explain, 255)
    bkgnd3 = gaussian(image_to_explain, 8, channel_axis=-1) * 255
    bkgnd4 = np.clip(np.random.normal(128, 128, size=image_to_explain.shape), 0, 255).astype(np.uint8)
    bkgnd4 = (gaussian(bkgnd4, 2.0, channel_axis=-1) * 255).astype(np.uint8)

    if bg_type == "black":
        background_image_set = np.array([bkgnd0])
    elif bg_type == "gray":
        background_image_set = np.array([bkgnd1])
    elif bg_type == "white":
        background_image_set = np.array([bkgnd2])
    elif bg_type == "blurred":
        background_image_set = np.array([bkgnd3])
    elif bg_type == "noise":
        background_image_set = np.array([bkgnd4])
    elif bg_type == "full":
        background_image_set = np.array([bkgnd0, bkgnd1, bkgnd2, bkgnd3, bkgnd4])
    else:
        raise ValueError(f"Unknown bg_type: {bg_type}")

    model_preprocess = transforms.Compose([transforms.ToTensor()])
    background_tensors = torch.cat(
        [torch.unsqueeze(model_preprocess(bkgnd.astype(np.float32) / 255.0), dim=0) for bkgnd in background_image_set]
    ).to(device)
    return image_to_explain, image_to_explain.copy(), background_image_set, background_tensors


def yolo_class_count(model: YOLO) -> int:
    names = getattr(model, "names", {})
    return len(names) if names is not None else 80


def yolo_result_to_scores(result, num_classes: int, aggregate: str = "sum"):
    p = np.zeros(num_classes)
    for cls, prob in zip(result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
        cls = int(cls)
        if cls >= num_classes:
            continue
        if aggregate == "max":
            p[cls] = max(p[cls], float(prob))
        else:
            p[cls] += float(prob)
    return p


def predict_yolo(model: YOLO, x, num_classes: int | None = None, verbose: bool = False, aggregate: str = "sum"):
    num_classes = yolo_class_count(model) if num_classes is None else num_classes
    result = model.predict(source=x, verbose=verbose)[0]
    return yolo_result_to_scores(result, num_classes=num_classes, aggregate=aggregate)


def predict_yolo_batch(model: YOLO, images, num_classes: int | None = None, verbose: bool = False, aggregate: str = "sum"):
    num_classes = yolo_class_count(model) if num_classes is None else num_classes
    images = list(images)
    if not images:
        return np.empty((0, num_classes))
    results = model.predict(source=images, verbose=verbose, batch=len(images))
    return np.stack(
        [yolo_result_to_scores(result, num_classes=num_classes, aggregate=aggregate) for result in results],
        axis=0,
    )


def make_predict_yolo_masked(model: YOLO, image_to_explain, background_image_set, aggregate: str = "sum"):
    num_classes = yolo_class_count(model)

    def predict_yolo_masked(masks, verbose: bool = False):
        masked_images = []
        mask_indexes = []
        for mask_index, mask in enumerate(masks):
            if len(mask.shape) == 2:
                mask3 = np.stack([mask, mask, mask], axis=2)
            else:
                mask3 = mask.copy()
            for repl in background_image_set:
                masked_image = np.where(mask3, image_to_explain, repl)
                masked_images.append(masked_image)
                mask_indexes.append(mask_index)

        batch_preds = predict_yolo_batch(model, masked_images, num_classes=num_classes, verbose=verbose, aggregate=aggregate)
        preds = np.zeros((len(masks), batch_preds.shape[1]))
        counts = np.zeros(len(masks))
        for mask_index, pred in zip(mask_indexes, batch_preds):
            preds[mask_index] += pred
            counts[mask_index] += 1
        return preds / counts[:, None]

    return predict_yolo_masked


def load_image_to_explain(image_id: str, image_dir: str, model: YOLO, class_names, device, bg_type: str, aggregate: str):
    image_path = Path(image_dir) / f"{image_id}.jpg"
    image, image_tensor, background_image_set, background_tensors = load_image(image_path, device=device, bg_type=bg_type)
    num_classes = yolo_class_count(model)
    predicted_fS = predict_yolo(model, image, num_classes=num_classes, aggregate=aggregate)
    sorted_classes = np.flip(np.argsort(predicted_fS))
    sorted_probs = predicted_fS[sorted_classes]
    predicted_cls = int(sorted_classes[0])
    predicted_f0 = predict_yolo_batch(model, background_image_set, num_classes=num_classes, aggregate=aggregate)
    predicted_f0 = np.mean(predicted_f0, axis=0)
    return {
        "fname": image_id,
        "image_to_explain": image,
        "image_to_explain_tensor": image_tensor,
        "background_image_set": background_image_set,
        "background_tensors": background_tensors,
        "predicted_fS": predicted_fS,
        "sorted_classes": sorted_classes,
        "sorted_probs": sorted_probs,
        "predicted_cls": predicted_cls,
        "fixed_category": class_names[predicted_cls],
        "f_S": float(predicted_fS[predicted_cls]),
        "predicted_f0": predicted_f0,
        "f_0": float(predicted_f0[predicted_cls]),
    }


def image_has_segmentation(coco: COCO, image_no: int) -> bool:
    image_info = coco.loadImgs(image_no)[0]
    ann_ids = coco.getAnnIds(imgIds=image_info["id"])
    annotations = coco.loadAnns(ann_ids)
    return any("segmentation" in ann for ann in annotations)


def collect_image_ids(masks_base_path: Path) -> list[str]:
    image_ids = [p.name.split("_")[0] for p in masks_base_path.glob("*_refined.npy")]
    return sorted(set(normalize_image_id(image_id) for image_id in image_ids))


def build_partition_summary(image_ids, masks_base_path: Path, path_results: Path, uts2):
    rows = []
    for image_id in image_ids:
        partition_results = uts2.load_json_partitions(str(masks_base_path), image_id, verbose=False)
        if not partition_results:
            continue
        rows.append(
            {
                "image_id": normalize_image_id(partition_results.get("name", image_id).split(".")[0]),
                "n_sam_masks": partition_results.get("n_sam_masks", partition_results.get("n_masks", 0)),
                "n_coverage_masks": partition_results.get("n_coverage_masks", 0),
                "n_refined_instances": partition_results.get("n_refined_instances", 0),
                "n_unique_filler": partition_results.get("n_unique_filler", 0),
                "total_time_sec": partition_results.get("total_time_sec", 0.0),
                "time_sam": partition_results.get("time_sam", 0.0),
                "time_coverage": partition_results.get("time_coverage", 0.0),
                "time_compact": partition_results.get("time_compact", 0.0),
                "time_filler": partition_results.get("time_filler", 0.0),
                "time_refined": partition_results.get("time_refined", 0.0),
                "steps": partition_results.get("steps", 0),
                "non_zero_px": partition_results.get("non_zero_px", 0),
                "total_px": partition_results.get("total_px", 0),
                "coverage": partition_results.get("coverage", 0.0),
            }
        )
    df = pd.DataFrame(rows)
    out_path = path_results / "partition_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"Partition summary saved to: {out_path}")
    return df


def save_current_auc_table(auc_table_rows: list[dict], path_results: Path) -> tuple[Path, pd.DataFrame]:
    auc_all_df = pd.DataFrame(auc_table_rows)
    current_run_path = path_results / "auc_results_all_images_current_run.csv"
    tmp_path = current_run_path.with_name(current_run_path.name + ".tmp")
    auc_all_df.to_csv(tmp_path, index=False)
    tmp_path.replace(current_run_path)
    return current_run_path, auc_all_df


def aggregate_auc_outputs(auc_all_df: pd.DataFrame, partition_summary_df: pd.DataFrame, path_results: Path, max_evals: int):
    no_images = len(auc_all_df["image_no"].unique())
    auc_all = auc_all_df.copy()
    auc_all["method"] = pd.Categorical(auc_all["method"], categories=METHOD_ORDER, ordered=True)
    auc_all = auc_all.sort_values(["method", "image_no"]).reset_index(drop=True)
    summary = (
        auc_all.groupby("method", observed=True)
        .agg(
            images=("image_no", "nunique"),
            auc_ins_mean=("auc_ins", "mean"),
            auc_ins_std=("auc_ins", "std"),
            auc_ins_median=("auc_ins", "median"),
            auc_del_mean=("auc_del", "mean"),
            auc_del_std=("auc_del", "std"),
            auc_del_median=("auc_del", "median"),
        )
        .reset_index()
    )

    auc_all_path = path_results / f"auc_results_all_images_{max_evals}_{no_images}.csv"
    auc_summary_path = path_results / f"auc_results_summary_{max_evals}_{no_images}.csv"
    box_plot_path = path_results / f"auc_results_boxplots_{max_evals}_{no_images}.png"
    auc_all.to_csv(auc_all_path, index=False)
    summary.to_csv(auc_summary_path, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)
    methods = [m for m in METHOD_ORDER if m in set(auc_all["method"].dropna().astype(str))]
    for ax, metric, title, ylabel in [
        (axes[0], "auc_ins", "$\\mathit{AUC}^{+}$ across images", "Higher is better"),
        (axes[1], "auc_del", "$\\mathit{AUC}^{-}$ across images", "Lower is better"),
    ]:
        values = [auc_all.loc[auc_all["method"].astype(str) == method, metric].dropna().values for method in methods]
        ax.boxplot(values, labels=methods, showmeans=True, patch_artist=True)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)

    partition_time_cols = [
        ("time_sam", "SAM"),
        ("time_coverage", "Coverage"),
        ("time_compact", "Compact"),
        ("time_filler", "Filler"),
        ("time_refined", "Refined"),
    ]
    partition_time_cols = [(col, label) for col, label in partition_time_cols if col in partition_summary_df.columns]
    values = [partition_summary_df[col].dropna().astype(float).values for col, _ in partition_time_cols]
    labels = [label for _, label in partition_time_cols]
    axes[2].boxplot(values, labels=labels, showmeans=True, patch_artist=True)
    axes[2].set_title("Time comparison for Partition only")
    axes[2].set_ylabel("Seconds (lower is better)")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].tick_params(axis="x", rotation=25)
    plt.suptitle(f"Aggregate AUC and Partition Time Across {no_images} Images & Budget: {max_evals}", fontsize=16)
    plt.tight_layout()
    plt.savefig(box_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved aggregate AUC table: {auc_all_path}")
    print(f"Saved aggregate summary: {auc_summary_path}")
    print(f"Saved box plot: {box_plot_path}")
    return auc_all, summary


def save_time_comparison_outputs(auc_all_df: pd.DataFrame, partition_summary_df: pd.DataFrame, path_results: Path, max_evals: int):
    no_images = len(auc_all_df["image_no"].unique())
    time_plot_path = path_results / f"time_comparison_partition_explanation_{max_evals}_{no_images}.png"
    time_summary_path = path_results / f"time_comparison_summary_{max_evals}_{no_images}.csv"
    time_auc_df = auc_all_df.copy()
    time_auc_df["method"] = pd.Categorical(time_auc_df["method"], categories=METHOD_ORDER, ordered=True)
    time_auc_df = time_auc_df.sort_values(["method", "image_no"]).reset_index(drop=True)
    methods = [method for method in METHOD_ORDER if method in set(time_auc_df["method"].dropna().astype(str))]

    partition_time_cols = [
        ("time_sam", "SAM"),
        ("time_coverage", "Coverage"),
        ("time_compact", "Compact"),
        ("time_filler", "Filler"),
        ("time_refined", "Refined"),
    ]
    partition_time_cols = [(col, label) for col, label in partition_time_cols if col in partition_summary_df.columns]
    partition_values = [partition_summary_df[col].dropna().astype(float).values for col, _ in partition_time_cols]
    partition_labels = [label for _, label in partition_time_cols]
    exp_values = [
        time_auc_df.loc[time_auc_df["method"].astype(str) == method, "time_exp"].dropna().astype(float).values
        for method in methods
    ]
    eval_values = [
        time_auc_df.loc[time_auc_df["method"].astype(str) == method, "time_eval"].dropna().astype(float).values
        for method in methods
    ]

    partition_time_summary = (
        partition_summary_df[[col for col, _ in partition_time_cols]]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .T.reset_index()
        .rename(columns={"index": "time_name"})
    )
    partition_time_summary["group"] = "partition"
    partition_time_summary["time_name"] = partition_time_summary["time_name"].map(dict(partition_time_cols))
    explanation_time_summary = (
        time_auc_df.groupby("method", observed=True)
        .agg(
            images=("image_no", "nunique"),
            time_exp_mean=("time_exp", "mean"),
            time_exp_std=("time_exp", "std"),
            time_exp_median=("time_exp", "median"),
            time_eval_mean=("time_eval", "mean"),
            time_eval_std=("time_eval", "std"),
            time_eval_median=("time_eval", "median"),
        )
        .reset_index()
    )
    partition_time_summary.to_csv(time_summary_path.with_name(time_summary_path.stem + "_partition.csv"), index=False)
    explanation_time_summary.to_csv(time_summary_path.with_name(time_summary_path.stem + "_explanation.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4), sharey=False)
    axes[0].boxplot(partition_values, labels=partition_labels, showmeans=True, patch_artist=True)
    axes[0].set_title("Partition time only")
    axes[1].boxplot(exp_values, labels=methods, showmeans=True, patch_artist=True)
    axes[1].set_title("Explanation compute time")
    axes[2].boxplot(eval_values, labels=methods, showmeans=True, patch_artist=True)
    axes[2].set_title("AUC evaluation time")
    for ax in axes:
        ax.set_ylabel("Seconds (lower is better)")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
    plt.suptitle(f"Time Comparison Across {no_images} Images & Budget: {max_evals}", fontsize=16)
    plt.tight_layout()
    plt.savefig(time_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved time comparison plot: {time_plot_path}")


def save_total_time_outputs(auc_all_df: pd.DataFrame, partition_summary_df: pd.DataFrame, path_results: Path, max_evals: int):
    no_images = len(auc_all_df["image_no"].unique())
    table_path = path_results / f"total_time_by_method_{max_evals}_{no_images}.csv"
    plot_path = path_results / f"total_time_by_method_{max_evals}_{no_images}.png"
    auc_df = auc_all_df.copy()
    auc_df["image_id_str"] = auc_df["image_id"].apply(normalize_image_id)
    partition_summary_df = partition_summary_df.copy()
    partition_summary_df["image_id_str"] = partition_summary_df["image_id"].apply(normalize_image_id)
    auc_df["method"] = pd.Categorical(auc_df["method"], categories=METHOD_ORDER, ordered=True)

    rows = []
    for method in METHOD_ORDER:
        method_df = auc_df[auc_df["method"].astype(str) == method].copy()
        if method_df.empty:
            continue
        partition_col = PARTITION_TIME_BY_METHOD.get(method)
        method_df["partition_time_sec"] = 0.0
        if partition_col is not None and partition_col in partition_summary_df.columns:
            method_df = method_df.merge(partition_summary_df[["image_id_str", partition_col]], on="image_id_str", how="left")
            method_df["partition_time_sec"] = method_df[partition_col].fillna(0).astype(float)
        method_df["explanation_time_sec"] = method_df["time_exp"].fillna(0).astype(float)
        method_df["total_time_sec"] = method_df["partition_time_sec"] + method_df["explanation_time_sec"]
        rows.append(
            {
                "method": method,
                "images": method_df["image_id_str"].nunique(),
                "partition_time_mean": method_df["partition_time_sec"].mean(),
                "partition_time_sum": method_df["partition_time_sec"].sum(),
                "explanation_time_mean": method_df["explanation_time_sec"].mean(),
                "explanation_time_sum": method_df["explanation_time_sec"].sum(),
                "total_time_mean": method_df["total_time_sec"].mean(),
                "total_time_sum": method_df["total_time_sec"].sum(),
                "total_time_median": method_df["total_time_sec"].median(),
            }
        )

    total_time = pd.DataFrame(rows)
    total_time.to_csv(table_path, index=False)
    x = np.arange(len(total_time))
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.bar(x, total_time["partition_time_sum"], label="Partition generation", color="dodgerblue", alpha=0.85)
    ax.bar(
        x,
        total_time["explanation_time_sum"],
        bottom=total_time["partition_time_sum"],
        label="Explanation",
        color="coral",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(total_time["method"].astype(str), rotation=25)
    ax.set_ylabel("Total seconds across images")
    ax.set_title("Total time by method: partition generation + explanation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved total time table: {table_path}")
    print(f"Saved total time plot: {plot_path}")


def build_xai_results_webpage(xai_results_root: Path):
    def escape(value):
        return html.escape(str(value))

    def fmt_float(value, digits=4):
        if value is None:
            return ""
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return escape(value)

    def rel_path(path):
        return escape(path.relative_to(xai_results_root).as_posix())

    def image_tag(path, label):
        if path is not None and path.exists():
            return f'<img src="{rel_path(path)}" alt="{escape(label)}" loading="lazy">'
        missing = path.name if path is not None else "not found"
        return f'<div class="missing">Missing image<br>{escape(missing)}</div>'

    def latest_matching_file(folder, pattern):
        matches = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0] if matches else None

    def fmt_table_value(value, column=None):
        if value is None or pd.isna(value):
            return ""
        if column in {"auc_ins", "auc_del", "f_S", "f_0"}:
            return fmt_float(value, 4)
        if column in {"time_exp", "time_eval"}:
            return fmt_float(value, 2)
        if column in {"image_no", "image_id"}:
            try:
                return str(int(value))
            except (TypeError, ValueError):
                return escape(value)
        return escape(value)

    def dataframe_to_html_table(df, columns=None):
        if df is None or df.empty:
            return '<p class="muted">No rows available.</p>'
        view = df.copy()
        if columns is not None:
            view = view[[col for col in columns if col in view.columns]]
        header = "".join(f"<th>{escape(col)}</th>" for col in view.columns)
        rows = []
        for _, row in view.iterrows():
            rows.append("<tr>" + "".join(f"<td>{fmt_table_value(row[col], col)}</td>" for col in view.columns) + "</tr>")
        return f'<table><thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table>'

    auc_all_path = latest_matching_file(xai_results_root, "auc_results_all_images_*.csv")
    auc_summary_path = latest_matching_file(xai_results_root, "auc_results_summary_*.csv")
    auc_boxplot_path = latest_matching_file(xai_results_root, "auc_results_boxplots_*.png")
    auc_all_df = pd.read_csv(auc_all_path) if auc_all_path is not None else pd.DataFrame()
    auc_summary_df = pd.read_csv(auc_summary_path) if auc_summary_path is not None else pd.DataFrame()
    if not auc_all_df.empty:
        auc_all_df["image_id_str"] = auc_all_df["image_id"].apply(normalize_image_id)

    def auc_results_table(image_id):
        if auc_all_df.empty:
            return '<p class="muted">No aggregate AUC table found.</p>'
        sub = auc_all_df.loc[auc_all_df["image_id_str"] == image_id].copy()
        if sub.empty:
            return '<p class="muted">No AUC rows found for this image.</p>'
        sub["method"] = pd.Categorical(sub["method"], categories=METHOD_ORDER, ordered=True)
        sub = sub.sort_values("method")
        return dataframe_to_html_table(sub, ["method", "auc_ins", "auc_del", "time_exp", "time_eval"])

    def top_classes_table(top_classes):
        if not top_classes:
            return '<p class="muted">No top-k classes saved.</p>'
        rows = []
        for rank, item in enumerate(top_classes, start=1):
            rows.append(
                "<tr>"
                f"<td>{rank}</td>"
                f"<td>{escape(item.get('class_id', ''))}</td>"
                f"<td>{escape(item.get('class_name', ''))}</td>"
                f"<td>{fmt_float(item.get('confidence'), 4)}</td>"
                "</tr>"
            )
        return '<table><thead><tr><th>Rank</th><th>Class ID</th><th>Class</th><th>Confidence</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>"

    def speed_table(speed):
        if not isinstance(speed, dict) or not speed:
            return '<p class="muted">No speed values saved.</p>'
        rows = "".join(f"<tr><td>{escape(k)}</td><td>{fmt_float(v, 3)}</td></tr>" for k, v in speed.items())
        return "<table><thead><tr><th>Stage</th><th>ms</th></tr></thead><tbody>" + rows + "</tbody></table>"

    items = []
    for pred_path in sorted(xai_results_root.glob("*/*_predictions.json")):
        image_dir = pred_path.parent
        pred = json.loads(pred_path.read_text())
        folder_id = image_dir.name
        image_id = normalize_image_id(pred.get("image_id", folder_id))
        xai_path = next(iter(sorted(image_dir.glob("*_xai_results.png"))), image_dir / f"{image_id}_xai_results.png")
        auc_path = next(iter(sorted(image_dir.glob("auc_results_*.png"))), image_dir / f"auc_results_{folder_id}.png")
        top_classes = pred.get("top_k_classes", []) or []
        top_class = top_classes[0] if top_classes else {}
        items.append(
            {
                "folder_id": folder_id,
                "image_id": image_id,
                "pred": pred,
                "xai_path": xai_path,
                "auc_path": auc_path,
                "summary": {
                    "image_id": image_id,
                    "fixed_category": pred.get("fixed_category", ""),
                    "explained_class": pred.get("explained_class", ""),
                    "top_class": top_class.get("class_name", ""),
                    "top_confidence": top_class.get("confidence", None),
                    "f_S": pred.get("f_S", None),
                    "f_0": pred.get("f_0", None),
                    "has_segmentation": pred.get("has_segmentation", ""),
                },
            }
        )
    if not items:
        raise FileNotFoundError(f"No *_predictions.json files found below: {xai_results_root}")

    auc_columns = ["image_no", "image_id", "fixed_category", "method", "auc_ins", "auc_del", "time_exp", "time_eval"]
    auc_top_html = f"""
<section class="top-card">
  <h2>Aggregate AUC Results</h2>
  <p class="muted">AUC table: {escape(auc_all_path.name if auc_all_path is not None else 'not found')}</p>
  <p class="muted">AUC summary: {escape(auc_summary_path.name if auc_summary_path is not None else 'not found')}</p>
  <figure>
    <figcaption>Aggregate AUC box plot</figcaption>
    {image_tag(auc_boxplot_path, 'Aggregate AUC box plot')}
  </figure>
  <h3>Aggregate Summary</h3>
  <div class="table-scroll small-scroll">{dataframe_to_html_table(auc_summary_df)}</div>
  <details class="auc-details">
    <summary>Full AUC Table</summary>
    <div class="table-scroll full-auc-table">{dataframe_to_html_table(auc_all_df, auc_columns)}</div>
  </details>
</section>"""

    overview_rows = []
    sections = []
    for item in items:
        pred = item["pred"]
        s = item["summary"]
        overview_rows.append(
            f'<tr data-image="{escape(item["image_id"])}" data-folder="{escape(item["folder_id"])}">'
            f'<td><a href="#{escape(item["image_id"])}">{escape(item["image_id"])}</a></td>'
            f'<td>{escape(s["fixed_category"])}</td>'
            f'<td>{escape(s["explained_class"])}</td>'
            f'<td>{escape(s["top_class"])}</td>'
            f'<td>{fmt_float(s["top_confidence"], 4)}</td>'
            f'<td>{fmt_float(s["f_S"], 4)}</td>'
            f'<td>{fmt_float(s["f_0"], 4)}</td>'
            f'<td>{escape(s["has_segmentation"])}</td>'
            "</tr>"
        )
        detail_rows = "".join(f"<tr><td>{escape(k)}</td><td>{escape(v)}</td></tr>" for k, v in s.items())
        sections.append(
            f"""
<section class="result-card" id="{escape(item['image_id'])}" data-image="{escape(item['image_id'])}" data-folder="{escape(item['folder_id'])}">
  <div class="section-title">
    <h2>Image {escape(item['image_id'])}</h2>
    <a href="#{escape(item['image_id'])}">#{escape(item['folder_id'])}</a>
  </div>
  <details class="prediction-details">
    <summary>Prediction details</summary>
    <div class="details-grid">
      <div><h3>Prediction Details</h3><table><tbody>{detail_rows}</tbody></table></div>
      <div><h3>Top-k Classes</h3>{top_classes_table(pred.get('top_k_classes', []))}</div>
      <div><h3>YOLO Speed</h3>{speed_table(pred.get('speed', {}))}</div>
    </div>
  </details>
  <details class="auc-details auc-image-table">
    <summary>AUC results for this image</summary>
    {auc_results_table(item['image_id'])}
  </details>
  <figure><figcaption>XAI results</figcaption>{image_tag(item['xai_path'], item['image_id'] + ' XAI results')}</figure>
  <figure><figcaption>AUC curves</figcaption>{image_tag(item['auc_path'], item['image_id'] + ' AUC curves')}</figure>
</section>"""
        )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>XAI Results Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #1f2933; background: #f7f8fa; }}
  header {{ padding: 24px 32px; background: #111827; color: white; }}
  h1, h2, h3 {{ margin: 0; }}
  header p {{ margin: 8px 0 0; color: #cbd5e1; }}
  main {{ padding: 24px 32px 48px; }}
  .controls {{ margin: 18px 0; }}
  input {{ width: min(460px, 100%); padding: 10px 12px; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 14px; }}
  .overview-wrap {{ max-height: 420px; overflow: auto; border: 1px solid #e5e7eb; border-radius: 8px; background: white; }}
  .table-scroll {{ overflow: auto; border: 1px solid #e5e7eb; border-radius: 8px; background: white; margin: 10px 0 18px; }}
  .small-scroll {{ max-height: 260px; }}
  .full-auc-table {{ max-height: 520px; }}
  table {{ border-collapse: collapse; width: 100%; background: white; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; vertical-align: top; }}
  th {{ background: #f1f5f9; position: sticky; top: 0; z-index: 1; }}
  a {{ color: #2563eb; text-decoration: none; }}
  .top-card, .result-card {{ margin-top: 24px; padding: 18px; background: white; border: 1px solid #e5e7eb; border-radius: 8px; }}
  .section-title {{ display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 14px; }}
  .prediction-details, .auc-details {{ margin-bottom: 16px; border: 1px solid #e5e7eb; border-radius: 8px; background: #fbfdff; }}
  .prediction-details summary, .auc-details summary {{ cursor: pointer; padding: 10px 12px; font-weight: 700; color: #334155; }}
  .details-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; padding: 0 12px 12px; }}
  .details-grid h3 {{ margin-bottom: 8px; font-size: 15px; }}
  .auc-details .table-scroll, .auc-image-table table {{ margin: 0 12px 12px; width: calc(100% - 24px); }}
  .top-card h3 {{ margin: 12px 0 8px; font-size: 15px; }}
  figure {{ margin: 0 0 16px; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; background: #fff; }}
  figcaption {{ padding: 8px 10px; font-weight: 700; background: #f8fafc; border-bottom: 1px solid #e5e7eb; }}
  img {{ display: block; width: 70%; height: auto; }}
  .missing {{ padding: 32px; text-align: center; color: #b91c1c; background: #fff1f2; }}
  .muted {{ color: #64748b; }}
</style>
</head>
<body>
<header><h1>XAI Results Report</h1><p>{len(items)} images from {escape(xai_results_root)}</p></header>
<main>
  <div class="controls"><input id="imageFilter" placeholder="Filter by image id, folder id, or class..." oninput="filterImages()"></div>
  {auc_top_html}
  <h2>All Images</h2>
  <div class="overview-wrap">
    <table>
      <thead><tr><th>Image</th><th>Fixed category</th><th>Explained class</th><th>Top class</th><th>Top confidence</th><th>f_S</th><th>f_0</th><th>Segmentation</th></tr></thead>
      <tbody>{"".join(overview_rows)}</tbody>
    </table>
  </div>
  {"".join(sections)}
</main>
<script>
function filterImages() {{
  const q = document.getElementById('imageFilter').value.trim().toLowerCase();
  document.querySelectorAll('[data-image]').forEach(el => {{
    const text = el.innerText.toLowerCase() + ' ' + el.dataset.image + ' ' + el.dataset.folder;
    el.style.display = text.includes(q) ? '' : 'none';
  }});
}}
</script>
</body>
</html>
"""
    html_report_path = xai_results_root / "xai_results_report.html"
    html_report_path.write_text(html_doc, encoding="utf-8")
    print(f"HTML report written to: {html_report_path}")
    return html_report_path


def run(args):
    project_root = find_project_root()
    scripts_dir = project_root / "examples" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import utils_sam as uts
    import utils_sam_2 as uts2
    import utils_xai as utx

    config = load_config(project_root, args.config)
    if args.output_dir:
        config.setdefault("output", {})
        config["output"]["dir"] = str(Path(args.output_dir).expanduser())
        config["output"]["folder"] = ""

    device = select_device(args.device or config.get("device", "auto"))
    checkpoint_dir = resolve_checkpoint_dir(project_root, args.checkpoint_dir)
    yolo_source = resolve_yolo_source(project_root, args.model, args.weights, checkpoint_dir=str(checkpoint_dir))
    model = YOLO(yolo_source)
    class_names = model.names
    print(f"Project root: {project_root}")
    print(f"Device: {device}")
    print(f"YOLO model: {args.model}")
    print(f"YOLO source: {yolo_source}")
    print(f"YOLO checkpoint dir: {checkpoint_dir}")
    print(f"YOLO classes: {yolo_class_count(model)}")
    print(f"shap_bpt version: {shap_bpt.__version__}")
    resource_logging = args.verbose_level in {"medium", "high"}
    if resource_logging:
        print(
            "Run settings: "
            f"max_evals={args.max_evals}, "
            f"explain_batch_size={args.eval_batch_size}, "
            f"auc_eval_batch_size={args.auc_batch_size}, "
            f"methods={','.join(METHOD_ORDER)}, "
            f"num_explained_classes={args.num_explained_classes}, "
            f"class_aggregation={args.class_aggregation}, "
            f"bg_type={args.bg_type}, "
            f"verbose_k={args.verbose_k}"
        )
        describe_compute_resources(device)

    masks_base_path = Path(config["data"]["masks_path"]) / config["data"]["mask_dir"] / config["data"]["mask_dir_final"]
    image_dir = config["data"]["image_dir"]
    path_results = Path(config["output"]["dir"]) / config["output"].get("folder", "") / "xai_results"
    path_results.mkdir(parents=True, exist_ok=True)
    coco = COCO(config["data"]["annotation_file"])

    image_ids = [normalize_image_id(x) for x in args.image_ids] if args.image_ids else collect_image_ids(masks_base_path)
    if args.limit is not None:
        image_ids = image_ids[: args.limit]
    print(f"Images to process: {len(image_ids)}")
    print(f"Masks: {masks_base_path}")
    print(f"Results: {path_results}")

    partition_summary_df = build_partition_summary(image_ids, masks_base_path, path_results, uts2)

    failures = []
    auc_table_rows = []
    top_classes_n = 5
    iteration_durations = []
    method_progress = args.verbose_level in {"medium", "high"}
    for iteration_index, image_id in enumerate(tqdm(image_ids, desc="Full image-set XAI", position=0, dynamic_ncols=True), start=1):
        iteration_start = time.time()
        first_iteration_breakdown = {
            "load_image_and_target": 0.0,
            "segmentation_lookup": 0.0,
            "yolo_predict": 0.0,
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
            input_data = load_image_to_explain(
                image_id,
                image_dir,
                model=model,
                class_names=class_names,
                device=device,
                bg_type=args.bg_type,
                aggregate=args.class_aggregation,
            )
            first_iteration_breakdown["load_image_and_target"] += time.time() - timing_start
            fixed_category = input_data["fixed_category"]
            timing_start = time.time()
            has_segmentation = image_has_segmentation(coco, image_no)
            first_iteration_breakdown["segmentation_lookup"] += time.time() - timing_start
            timing_start = time.time()
            results = model.predict(input_data["image_to_explain"], verbose=False)
            first_iteration_breakdown["yolo_predict"] += time.time() - timing_start
            top_k_classes = utx.get_top_k_classes(results, model.names, k=top_classes_n)
            path_results_img = path_results / str(image_no)
            path_results_img.mkdir(parents=True, exist_ok=True)

            predict_masked = make_predict_yolo_masked(
                model,
                input_data["image_to_explain"],
                input_data["background_image_set"],
                aggregate=args.class_aggregation,
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
            input_data["explained_class"] = class_names[predicted_cls]
            input_data["f_S"] = f_S
            input_data["f_0"] = f_0
            timing_start = time.time()
            utx.save_yolo_predictions(results, input_data, config, has_segmentation, top_k_classes)
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
                utx.plot_xai(
                    input_data,
                    partitions,
                    shap_values,
                    model,
                    results,
                    image_id,
                    destroy_fig=True,
                    save_path=str(path_results_img),
                    save_fig=True,
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
                image_auc_rows = utx.auc_results_to_rows(
                    auc_results,
                    image_no=image_no,
                    image_id=image_id,
                    fixed_category=fixed_category,
                    f_S=f_S,
                    f_0=f_0,
                )
                auc_table_rows.extend(image_auc_rows)
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
                    f"yolo_pred={format_duration(first_iteration_breakdown['yolo_predict'])}",
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
    parser.add_argument("--model", default="yolo11s", help="YOLO model/checkpoint name. Examples: yolo11s, yolo_26.")
    parser.add_argument("--weights", default=None, help="Explicit YOLO weights path. Overrides --model.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for YOLO checkpoint cache. Default: examples/notebooks/checkpoints.",
    )
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"], help="Override config device.")
    parser.add_argument("--output-dir", default=None, help="Override config output.dir; xai_results is created inside it.")
    parser.add_argument("--max-evals", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images for a smoke test.")
    parser.add_argument("--image-ids", nargs="*", default=None, help="Optional explicit image ids.")
    parser.add_argument("--bg-type", default="noise", choices=["black", "gray", "white", "blurred", "noise", "full"])
    parser.add_argument("--class-aggregation", default="sum", choices=["sum", "max"])
    parser.add_argument("--num-explained-classes", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--auc-batch-size", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--verbose-level", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--verbose-k", type=int, default=10, help="Print medium/high resource summary every K images after image 1.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-auc", dest="compute_auc", action="store_false")
    parser.add_argument("--no-plots", dest="save_plots", action="store_false")
    parser.add_argument("--no-html", dest="build_html", action="store_false")
    parser.set_defaults(compute_auc=True, save_plots=True, build_html=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
