#!/usr/bin/env python
"""Compare multiple full-image-set XAI experiment result folders.

The script loads saved AUC/faithfulness CSV files from folders such as
``xai_results_E1_1_2`` or ``xai_results_E3_1`` and creates aggregate comparison
tables, plots, and an HTML page.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["BPT", "sam", "coverage", "compact", "filled", "refined", "refined_filled"]

METRIC_DIRECTIONS = {
    "auc_ins": "higher",
    "auc_del": "lower",
    "drop_at_10": "higher",
    "drop_at_20": "higher",
    "drop_at_30": "higher",
    "insert_at_10": "higher",
    "insert_at_20": "higher",
    "insert_at_30": "higher",
    "sufficiency_at_10": "higher",
    "sufficiency_at_20": "higher",
    "sufficiency_gap_at_10": "lower",
    "sufficiency_gap_at_20": "lower",
    "comprehensiveness_at_10": "higher",
    "comprehensiveness_at_20": "higher",
    "sensitivity_n_corr": "higher",
    "stability_top10_jaccard": "higher",
    "total_time_mean": "lower",
}

DEFAULT_METRICS = [
    "auc_ins",
    "auc_del",
    "sensitivity_n_corr",
    "comprehensiveness_at_20",
    "sufficiency_gap_at_20",
    "total_time_mean",
]

EXP_MODEL_NAMES = {
    "E1": "YOLOv11",
    "E2": "ResNet-50",
    "E3": "ViT-B/16",
    "E4": "DETR",
}


def latest_file(folder: Path, pattern: str, exclude=()) -> Path | None:
    exclude = set(exclude)
    matches = [path for path in folder.glob(pattern) if path.name not in exclude]
    if not matches:
        return None
    return sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def infer_exp_no(results_dir: Path, run_info: dict | None = None) -> str:
    if run_info and run_info.get("exp_no"):
        return str(run_info["exp_no"])
    match = re.search(r"(E\d(?:_[A-Za-z0-9]+)+)", results_dir.name)
    return match.group(1) if match else results_dir.name


def infer_model_name(exp_no: str, run_info: dict | None = None) -> str:
    if run_info:
        if run_info.get("model_group") == "yolov11" or str(run_info.get("model", "")).lower().startswith("yolo"):
            return "YOLOv11"
        if run_info.get("model_type") == "resnet50" or "resnet" in str(run_info.get("model", "")).lower():
            return "ResNet-50"
        if run_info.get("model_type") == "vit_b_16" or "vit" in str(run_info.get("model", "")).lower():
            return "ViT-B/16"
        if run_info.get("model_group") == "detr" or "detr" in str(run_info.get("model", "")).lower():
            return "DETR"
    prefix = exp_no.split("_", 1)[0]
    return EXP_MODEL_NAMES.get(prefix, prefix)


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def resolve_auc_csv(results_dir: Path) -> Path:
    final_csv = latest_file(results_dir, "auc_results_all_images_*.csv", exclude={"auc_results_all_images_current_run.csv"})
    if final_csv is not None:
        return final_csv
    current_csv = results_dir / "auc_results_all_images_current_run.csv"
    if current_csv.exists():
        return current_csv
    raise FileNotFoundError(f"No AUC CSV found in {results_dir}")


def normalize_image_id(value) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(12)


def load_total_time(results_dir: Path) -> pd.DataFrame:
    total_path = latest_file(results_dir, "total_time_by_method_*.csv")
    if total_path is None:
        return pd.DataFrame(columns=["method", "total_time_mean", "total_time_sum"])
    df = pd.read_csv(total_path)
    keep = [col for col in ["method", "total_time_mean", "total_time_sum", "partition_time_mean", "explanation_time_mean"] if col in df.columns]
    return df[keep].copy()


def load_experiment(results_dir: Path) -> tuple[pd.DataFrame, dict]:
    results_dir = results_dir.expanduser().resolve()
    run_info = read_json(results_dir / "run_info.json") or {}
    exp_no = infer_exp_no(results_dir, run_info)
    model_name = infer_model_name(exp_no, run_info)
    auc_csv = resolve_auc_csv(results_dir)
    auc_df = pd.read_csv(auc_csv)
    auc_df["experiment"] = exp_no
    auc_df["model"] = model_name
    auc_df["results_dir"] = str(results_dir)
    auc_df["auc_csv"] = str(auc_csv)
    if "image_id" in auc_df.columns:
        auc_df["image_id_str"] = auc_df["image_id"].apply(normalize_image_id)
    metadata = {
        "experiment": exp_no,
        "model": model_name,
        "results_dir": str(results_dir),
        "auc_csv": str(auc_csv),
        "run_info": run_info,
    }
    return auc_df, metadata


def summarize_by_experiment_method(combined: pd.DataFrame, results_dirs: list[Path]) -> pd.DataFrame:
    metric_cols = [metric for metric in METRIC_DIRECTIONS if metric in combined.columns and metric != "total_time_mean"]
    agg_spec = {
        "images": ("image_no", "nunique") if "image_no" in combined.columns else ("image_id", "nunique"),
    }
    for metric in metric_cols:
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_std"] = (metric, "std")
        agg_spec[f"{metric}_median"] = (metric, "median")

    summary = (
        combined.groupby(["experiment", "model", "method"], dropna=False, observed=True)
        .agg(**agg_spec)
        .reset_index()
    )

    time_frames = []
    for results_dir in results_dirs:
        run_info = read_json(results_dir / "run_info.json") or {}
        exp_no = infer_exp_no(results_dir, run_info)
        model_name = infer_model_name(exp_no, run_info)
        time_df = load_total_time(results_dir)
        if time_df.empty:
            continue
        time_df["experiment"] = exp_no
        time_df["model"] = model_name
        time_frames.append(time_df)
    if time_frames:
        time_all = pd.concat(time_frames, ignore_index=True)
        summary = summary.merge(time_all, on=["experiment", "model", "method"], how="left")
    return summary


def normalize_metric(series: pd.Series, direction: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    min_value = values.min(skipna=True)
    max_value = values.max(skipna=True)
    if pd.isna(min_value) or pd.isna(max_value) or np.isclose(max_value, min_value):
        return pd.Series(np.nan, index=series.index)
    score = (values - min_value) / (max_value - min_value)
    if direction == "lower":
        score = 1.0 - score
    return score


def add_ranking_columns(summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    ranked = summary.copy()
    score_cols = []
    for metric in metrics:
        source_col = metric if metric == "total_time_mean" else f"{metric}_mean"
        if source_col not in ranked.columns:
            continue
        score_col = f"{metric}_score"
        ranked[score_col] = normalize_metric(ranked[source_col], METRIC_DIRECTIONS[metric])
        score_cols.append(score_col)
    if score_cols:
        ranked["composite_score"] = ranked[score_cols].mean(axis=1, skipna=True)
        ranked["composite_rank"] = ranked["composite_score"].rank(ascending=False, method="min")
    else:
        ranked["composite_score"] = np.nan
        ranked["composite_rank"] = np.nan
    return ranked.sort_values(["composite_rank", "experiment", "method"]).reset_index(drop=True)


def plot_metric_bars(ranked: pd.DataFrame, metrics: list[str], output_dir: Path) -> Path:
    available = []
    for metric in metrics:
        col = metric if metric == "total_time_mean" else f"{metric}_mean"
        if col in ranked.columns:
            available.append((metric, col))
    n_cols = 2
    n_rows = int(np.ceil(len(available) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.5 * n_cols, 4.2 * n_rows), squeeze=False)

    for ax, (metric, col) in zip(axes.flat, available):
        view = ranked[["experiment", "method", col]].dropna().copy()
        view["label"] = view["experiment"].astype(str) + " / " + view["method"].astype(str)
        view = view.sort_values(col, ascending=METRIC_DIRECTIONS[metric] == "lower")
        ax.barh(view["label"], view[col], color="#4f83cc", alpha=0.85)
        ax.invert_yaxis()
        ax.set_title(f"{metric} ({METRIC_DIRECTIONS[metric]} is better)")
        ax.grid(axis="x", alpha=0.25)

    for ax in axes.flat[len(available):]:
        ax.axis("off")
    plt.tight_layout()
    out_path = output_dir / "overall_metric_bars.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_composite_heatmap(ranked: pd.DataFrame, metrics: list[str], output_dir: Path) -> Path:
    score_cols = [f"{metric}_score" for metric in metrics if f"{metric}_score" in ranked.columns]
    labels = (ranked["experiment"].astype(str) + " / " + ranked["method"].astype(str)).tolist()
    data = ranked[score_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, len(score_cols) * 1.2), max(6, len(labels) * 0.28)))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(np.arange(len(score_cols)))
    ax.set_xticklabels([col.replace("_score", "") for col in score_cols], rotation=35, ha="right")
    ax.set_title("Normalized metric scores (1 is best)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    plt.tight_layout()
    out_path = output_dir / "overall_normalized_score_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_quality_time_scatter(ranked: pd.DataFrame, output_dir: Path) -> Path | None:
    if "total_time_mean" not in ranked.columns or "composite_score" not in ranked.columns:
        return None
    view = ranked.dropna(subset=["total_time_mean", "composite_score"]).copy()
    if view.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 6))
    for experiment, group in view.groupby("experiment"):
        ax.scatter(group["total_time_mean"], group["composite_score"], s=70, label=experiment, alpha=0.85)
        for _, row in group.iterrows():
            ax.annotate(str(row["method"]), (row["total_time_mean"], row["composite_score"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Mean total time per image (seconds, lower better)")
    ax.set_ylabel("Composite normalized score (higher better)")
    ax.set_title("Quality-time tradeoff")
    ax.grid(alpha=0.25)
    ax.legend(title="Experiment", fontsize=8)
    plt.tight_layout()
    out_path = output_dir / "quality_time_tradeoff.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def dataframe_to_html(df: pd.DataFrame, max_rows: int | None = None) -> str:
    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)
    return view.to_html(index=False, escape=True, border=0, classes="data-table")


def rel_path(path: Path | None, output_dir: Path) -> str:
    if path is None:
        return ""
    try:
        return html.escape(path.relative_to(output_dir).as_posix())
    except ValueError:
        return html.escape(str(path))


def build_html_report(ranked: pd.DataFrame, combined: pd.DataFrame, metadata: list[dict], image_paths: list[Path], output_dir: Path) -> Path:
    image_html = "\n".join(
        f"<figure><figcaption>{html.escape(path.stem)}</figcaption><img src='{rel_path(path, output_dir)}' alt='{html.escape(path.stem)}'></figure>"
        for path in image_paths
        if path is not None
    )
    metadata_df = pd.DataFrame([{k: v for k, v in item.items() if k != "run_info"} for item in metadata])
    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Experiment Comparison</title>
<style>
body {{ margin: 0; font-family: Arial, sans-serif; background: #f8fafc; color: #0f172a; }}
header {{ padding: 22px 28px; background: #111827; color: white; }}
main {{ padding: 22px 28px; }}
section {{ margin: 0 0 24px; padding: 16px; background: white; border: 1px solid #e5e7eb; border-radius: 8px; }}
.table-wrap {{ overflow: auto; max-height: 560px; border: 1px solid #e5e7eb; border-radius: 8px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; white-space: nowrap; }}
th {{ position: sticky; top: 0; background: #eef2ff; cursor: pointer; }}
figure {{ margin: 0 0 20px; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; }}
figcaption {{ padding: 8px 10px; font-weight: 700; background: #f1f5f9; }}
img {{ display: block; width: 100%; height: auto; }}
</style>
<script>
function sortTable(th) {{
  const table = th.closest('table');
  const tbody = table.tBodies[0];
  const index = Array.from(th.parentNode.children).indexOf(th);
  const asc = th.dataset.asc !== 'true';
  Array.from(table.querySelectorAll('th')).forEach(cell => cell.dataset.asc = '');
  th.dataset.asc = asc ? 'true' : 'false';
  const rows = Array.from(tbody.rows);
  rows.sort((a, b) => {{
    const av = a.cells[index].innerText.trim();
    const bv = b.cells[index].innerText.trim();
    const an = Number(av);
    const bn = Number(bv);
    if (!Number.isNaN(an) && !Number.isNaN(bn)) {{
      return asc ? an - bn : bn - an;
    }}
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  }});
  rows.forEach(row => tbody.appendChild(row));
}}
window.addEventListener('DOMContentLoaded', () => {{
  document.querySelectorAll('th').forEach(th => {{
    th.title = 'Click to sort';
    th.addEventListener('click', () => sortTable(th));
  }});
}});
</script>
</head>
<body>
<header>
  <h1>Experiment Comparison</h1>
  <p>{len(metadata)} experiments, {combined['image_no'].nunique() if 'image_no' in combined.columns else 'unknown'} unique image ids, {len(combined)} metric rows.</p>
</header>
<main>
  <section>
    <h2>Experiments</h2>
    <div class="table-wrap">{dataframe_to_html(metadata_df)}</div>
  </section>
  <section>
    <h2>Overall Ranking</h2>
    <p>Composite score is the mean of normalized selected metrics. For lower-is-better metrics, the normalized score is inverted.</p>
    <div class="table-wrap">{dataframe_to_html(ranked, max_rows=100)}</div>
  </section>
  <section>
    <h2>Plots</h2>
    {image_html}
  </section>
</main>
</body>
</html>
"""
    out_path = output_dir / "experiment_comparison.html"
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path


def discover_results_dirs(results_root: Path, experiments: list[str] | None) -> list[Path]:
    if experiments:
        return [results_root / exp for exp in experiments]
    return sorted(path for path in results_root.glob("xai_results*") if path.is_dir())


def run(args):
    if args.results_dirs:
        results_dirs = [Path(path).expanduser().resolve() for path in args.results_dirs]
    else:
        results_root = Path(args.results_root).expanduser().resolve()
        results_dirs = discover_results_dirs(results_root, args.experiments)

    if not results_dirs:
        raise FileNotFoundError("No result directories were provided or discovered.")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else results_dirs[0].parent / "experiment_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, metadata = [], []
    for results_dir in results_dirs:
        auc_df, meta = load_experiment(results_dir)
        frames.append(auc_df)
        metadata.append(meta)
        print(f"Loaded {meta['experiment']}: {len(auc_df)} rows from {meta['auc_csv']}")

    combined = pd.concat(frames, ignore_index=True)
    if "method" in combined.columns:
        combined["method"] = pd.Categorical(combined["method"], categories=METHOD_ORDER, ordered=True)
        combined = combined.sort_values(["experiment", "method", "image_no" if "image_no" in combined.columns else "image_id"]).reset_index(drop=True)

    summary = summarize_by_experiment_method(combined, results_dirs)
    metrics = args.metrics or DEFAULT_METRICS
    ranked = add_ranking_columns(summary, metrics)

    combined_path = output_dir / "combined_auc_results.csv"
    summary_path = output_dir / "combined_method_summary.csv"
    ranked_path = output_dir / "combined_method_ranking.csv"
    combined.to_csv(combined_path, index=False)
    summary.to_csv(summary_path, index=False)
    ranked.to_csv(ranked_path, index=False)

    plot_paths = [
        plot_metric_bars(ranked, metrics, output_dir),
        plot_composite_heatmap(ranked, metrics, output_dir),
    ]
    quality_time_path = plot_quality_time_scatter(ranked, output_dir)
    if quality_time_path is not None:
        plot_paths.append(quality_time_path)

    report_path = build_html_report(ranked, combined, metadata, plot_paths, output_dir)

    print(f"Saved combined rows: {combined_path}")
    print(f"Saved method summary: {summary_path}")
    print(f"Saved method ranking: {ranked_path}")
    print(f"Saved HTML report: {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="examples/results", help="Folder containing xai_results_* directories.")
    parser.add_argument("--results-dirs", nargs="*", default=None, help="Explicit result directories to compare.")
    parser.add_argument("--experiments", nargs="*", default=None, help="Directory names under --results-root, e.g. xai_results_E1_1_2 xai_results_E3_1.")
    parser.add_argument("--output-dir", default=None, help="Where to save combined CSVs, plots, and HTML.")
    parser.add_argument("--metrics", nargs="*", default=None, choices=sorted(METRIC_DIRECTIONS), help="Metrics used for ranking and plots.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
