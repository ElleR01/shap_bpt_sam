#!/usr/bin/env python
"""Regenerate summary plots and HTML reports from saved YOLO + ShapBPT results."""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT_HINT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_HINT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_yolo_full import (  # noqa: E402
    aggregate_auc_outputs,
    build_xai_results_webpage,
    find_project_root,
    load_config,
    normalize_image_id,
    save_time_comparison_outputs,
    save_total_time_outputs,
)


def resolve_results_dir(project_root: Path, args) -> Path:
    if args.results_dir:
        path = Path(args.results_dir).expanduser()
        if not path.is_absolute():
            path = project_root / path
        return path

    config = load_config(project_root, args.config)
    path = Path(config["output"]["dir"]) / config["output"].get("folder", "") / "xai_results"
    if not path.is_absolute():
        path = project_root / path
    return path


def latest_file(folder: Path, pattern: str, exclude=()) -> Path | None:
    matches = [path for path in folder.glob(pattern) if path.name not in set(exclude)]
    if not matches:
        return None
    return sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def resolve_auc_csv(results_dir: Path, auc_csv: str | None = None) -> Path:
    if auc_csv:
        path = Path(auc_csv).expanduser()
        if not path.is_absolute():
            path = results_dir / path
        return path

    current_run = results_dir / "auc_results_all_images_current_run.csv"
    if current_run.exists():
        return current_run

    path = latest_file(results_dir, "auc_results_all_images_*.csv", exclude={"auc_results_all_images_current_run.csv"})
    if path is None:
        raise FileNotFoundError(f"No AUC table found in: {results_dir}")
    return path


def infer_max_evals(path: Path, fallback: int) -> int:
    match = re.search(r"auc_results_all_images_(\d+)_\d+\.csv$", path.name)
    if match:
        return int(match.group(1))
    return fallback


def resolve_partition_summary(results_dir: Path, partition_summary: str | None = None) -> pd.DataFrame:
    if partition_summary:
        path = Path(partition_summary).expanduser()
        if not path.is_absolute():
            path = results_dir / path
    else:
        path = results_dir / "partition_summary.csv"

    if not path.exists():
        print(f"Partition summary not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def prepare_auc_table(auc_all_df: pd.DataFrame) -> pd.DataFrame:
    auc_all_df = auc_all_df.copy()
    defaults = {
        "time_exp": 0.0,
        "time_eval": 0.0,
        "fixed_category": "",
        "f_S": None,
        "f_0": None,
    }
    for column, default in defaults.items():
        if column not in auc_all_df.columns:
            auc_all_df[column] = default
    return auc_all_df


def prepare_partition_summary(partition_summary_df: pd.DataFrame, auc_all_df: pd.DataFrame) -> pd.DataFrame:
    time_cols = ["time_sam", "time_coverage", "time_compact", "time_filler", "time_refined"]
    if partition_summary_df.empty or "image_id" not in partition_summary_df.columns:
        print("Using zero partition times because partition_summary.csv is unavailable.")
        image_ids = auc_all_df["image_id"].dropna().astype(str).unique() if "image_id" in auc_all_df.columns else []
        partition_summary_df = pd.DataFrame({"image_id": image_ids})
    else:
        partition_summary_df = partition_summary_df.copy()
    for column in time_cols:
        if column not in partition_summary_df.columns:
            partition_summary_df[column] = 0.0
    return partition_summary_df


def html_escape(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return html.escape(str(value))


def format_html_value(value, digits=4) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return html_escape(value)


def add_outlier_flags(auc_df: pd.DataFrame, metric: str, higher_is_better: bool, top_n: int) -> pd.DataFrame:
    df = auc_df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])
    if df.empty:
        return df

    flagged = []
    for method, group in df.groupby("method", dropna=False):
        q1 = group[metric].quantile(0.25)
        q3 = group[metric].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        method_outliers = group[group[metric] < lower] if higher_is_better else group[group[metric] > upper]
        method_outliers = method_outliers.copy()
        if not method_outliers.empty:
            method_outliers["outlier_reason"] = f"IQR by method={method}"
            flagged.append(method_outliers)

    worst = df.sort_values(metric, ascending=higher_is_better).head(top_n).copy()
    worst["outlier_reason"] = f"worst {min(top_n, len(worst))}"
    flagged.append(worst)

    out = pd.concat(flagged, ignore_index=True) if flagged else worst
    key_cols = [col for col in ["image_no", "image_id", "method"] if col in out.columns]
    if key_cols:
        out = out.drop_duplicates(key_cols, keep="first")
    out = out.sort_values(metric, ascending=higher_is_better).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def image_artifacts(results_dir: Path, row: pd.Series) -> tuple[Path | None, Path | None, str]:
    folder_id = ""
    if "image_no" in row and not pd.isna(row["image_no"]):
        try:
            folder_id = str(int(row["image_no"]))
        except (TypeError, ValueError):
            folder_id = str(row["image_no"])
    elif "image_id" in row and not pd.isna(row["image_id"]):
        try:
            folder_id = str(int(str(row["image_id"])))
        except (TypeError, ValueError):
            folder_id = str(row["image_id"])

    image_dir = results_dir / folder_id
    if not image_dir.exists() and "image_id" in row and not pd.isna(row["image_id"]):
        image_dir = results_dir / normalize_image_id(row["image_id"])

    xai_path = next(iter(sorted(image_dir.glob("*_xai_results.png"))), None) if image_dir.exists() else None
    auc_path = next(iter(sorted(image_dir.glob("auc_results_*.png"))), None) if image_dir.exists() else None
    return xai_path, auc_path, folder_id


def rel_path(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return html_escape(path.relative_to(root).as_posix())
    except ValueError:
        return html_escape(str(path))


def sortable_auc_table(df: pd.DataFrame, results_dir: Path, metric: str, caption: str) -> str:
    if df.empty:
        return f"<h2>{html_escape(caption)}</h2><p class='muted'>No rows found.</p>"

    columns = [
        "rank",
        "image_no",
        "image_id",
        "method",
        "fixed_category",
        "auc_ins",
        "auc_del",
        "time_exp",
        "time_eval",
        "outlier_reason",
    ]
    columns = [col for col in columns if col in df.columns]
    header = "".join(f"<th onclick=\"sortTable(this)\">{html_escape(col)}</th>" for col in columns)
    rows = []
    for _, row in df.iterrows():
        _, _, folder_id = image_artifacts(results_dir, row)
        cells = []
        for col in columns:
            value = row[col]
            if col in {"auc_ins", "auc_del", "time_exp", "time_eval"}:
                text = format_html_value(value, 4 if col.startswith("auc") else 2)
            elif col in {"image_no", "image_id"}:
                text = html_escape(value)
                if col == "image_no":
                    text = f"<a href='#image-{html_escape(folder_id)}'>{text}</a>"
            else:
                text = html_escape(value)
            cells.append(f"<td>{text}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        f"<section class='table-card'><h2>{html_escape(caption)}</h2>"
        f"<p class='muted'>Click any column header to sort. Initial order is worst {html_escape(metric)} first.</p>"
        f"<div class='table-scroll'><table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def build_auc_outlier_report(auc_all_df: pd.DataFrame, results_dir: Path, top_n: int = 50) -> Path:
    required = {"auc_ins", "auc_del"}
    missing = required - set(auc_all_df.columns)
    if missing:
        raise ValueError(f"AUC table is missing required columns: {sorted(missing)}")

    auc_df = prepare_auc_table(auc_all_df)
    auc_plus_outliers = add_outlier_flags(auc_df, "auc_ins", higher_is_better=True, top_n=top_n)
    auc_minus_outliers = add_outlier_flags(auc_df, "auc_del", higher_is_better=False, top_n=top_n)

    image_rows = pd.concat([auc_plus_outliers, auc_minus_outliers], ignore_index=True)
    image_cards = []
    seen = set()
    for _, row in image_rows.iterrows():
        xai_path, auc_path, folder_id = image_artifacts(results_dir, row)
        if not folder_id or folder_id in seen:
            continue
        seen.add(folder_id)
        image_id = normalize_image_id(row.get("image_id", folder_id))
        xai_html = f"<img src='{rel_path(xai_path, results_dir)}' alt='XAI {html_escape(image_id)}'>" if xai_path else "<div class='missing'>Missing XAI image</div>"
        auc_html = f"<img src='{rel_path(auc_path, results_dir)}' alt='AUC {html_escape(image_id)}'>" if auc_path else "<div class='missing'>Missing AUC image</div>"
        image_cards.append(
            f"<section class='image-card' id='image-{html_escape(folder_id)}'>"
            f"<h2>Image {html_escape(image_id)}</h2>"
            f"<figure><figcaption>XAI results</figcaption>{xai_html}</figure>"
            f"<figure><figcaption>AUC curves</figcaption>{auc_html}</figure>"
            "</section>"
        )

    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>AUC Outlier Report</title>
<style>
body {{ margin: 0; font-family: Arial, sans-serif; background: #f8fafc; color: #0f172a; }}
header {{ padding: 22px 28px; background: #111827; color: white; }}
main {{ padding: 22px 28px; }}
.muted {{ color: #64748b; }}
.table-card, .image-card {{ margin: 0 0 24px; padding: 16px; background: white; border: 1px solid #e5e7eb; border-radius: 8px; }}
.table-scroll {{ max-height: 520px; overflow: auto; border: 1px solid #e5e7eb; border-radius: 8px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; }}
th {{ position: sticky; top: 0; background: #eef2ff; cursor: pointer; user-select: none; }}
a {{ color: #2563eb; text-decoration: none; font-weight: 700; }}
figure {{ margin: 0 0 16px; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; background: white; }}
figcaption {{ padding: 8px 10px; font-weight: 700; background: #f1f5f9; }}
img {{ display: block; width: 100%; max-width: 1200px; height: auto; }}
.missing {{ padding: 28px; color: #b91c1c; background: #fff1f2; }}
</style>
</head>
<body>
<header>
  <h1>AUC Outlier Report</h1>
  <p>{len(auc_df)} AUC rows, {auc_df['image_no'].nunique() if 'image_no' in auc_df.columns else 'unknown'} images</p>
</header>
<main>
  {sortable_auc_table(auc_plus_outliers, results_dir, 'AUC+', 'Worst / Outlier AUC+ Rows')}
  {sortable_auc_table(auc_minus_outliers, results_dir, 'AUC-', 'Worst / Outlier AUC- Rows')}
  <h1>Outlier Images</h1>
  {''.join(image_cards)}
</main>
<script>
function cellValue(row, index) {{
  return row.children[index].innerText.trim();
}}
function asNumber(value) {{
  const n = Number(value.replace(/,/g, ''));
  return Number.isNaN(n) ? null : n;
}}
function sortTable(th) {{
  const table = th.closest('table');
  const tbody = table.querySelector('tbody');
  const index = Array.from(th.parentNode.children).indexOf(th);
  const current = th.dataset.sort || 'desc';
  const next = current === 'asc' ? 'desc' : 'asc';
  th.dataset.sort = next;
  const rows = Array.from(tbody.querySelectorAll('tr'));
  rows.sort((a, b) => {{
    const av = cellValue(a, index);
    const bv = cellValue(b, index);
    const an = asNumber(av);
    const bn = asNumber(bv);
    let cmp = 0;
    if (an !== null && bn !== null) cmp = an - bn;
    else cmp = av.localeCompare(bv);
    return next === 'asc' ? cmp : -cmp;
  }});
  rows.forEach(row => tbody.appendChild(row));
}}
</script>
</body>
</html>
"""
    out_path = results_dir / "auc_outlier_report.html"
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"AUC outlier report written to: {out_path}")
    return out_path


def run(args):
    project_root = find_project_root()
    results_dir = resolve_results_dir(project_root, args)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not args.boxplots and not args.html_report and not args.outlier_report:
        args.boxplots = True
        args.html_report = True
        args.outlier_report = True

    print(f"Project root: {project_root}")
    print(f"Results dir: {results_dir}")

    auc_all_df = None
    if args.boxplots or args.outlier_report:
        auc_csv = resolve_auc_csv(results_dir, args.auc_csv)
        max_evals = infer_max_evals(auc_csv, args.max_evals)
        auc_all_df = prepare_auc_table(pd.read_csv(auc_csv))
        print(f"AUC table: {auc_csv}")
        print(f"Images in AUC table: {auc_all_df['image_no'].nunique() if 'image_no' in auc_all_df.columns else 'unknown'}")
        print(f"Max evals: {max_evals}")

    if args.boxplots:
        partition_summary_df = prepare_partition_summary(resolve_partition_summary(results_dir, args.partition_summary), auc_all_df)
        aggregate_auc_outputs(auc_all_df, partition_summary_df, results_dir, max_evals)
        save_time_comparison_outputs(auc_all_df, partition_summary_df, results_dir, max_evals)
        save_total_time_outputs(auc_all_df, partition_summary_df, results_dir, max_evals)

    if args.html_report:
        report_path = build_xai_results_webpage(results_dir)
        print(f"HTML report: {report_path}")

    if args.outlier_report:
        outlier_path = build_auc_outlier_report(auc_all_df, results_dir, top_n=args.outlier_top_n)
        print(f"AUC outlier report: {outlier_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="MSCOCO_mac", help="Config name/path used only when --results-dir is omitted.")
    parser.add_argument("--results-dir", default=None, help="Path to xai_results folder.")
    parser.add_argument("--auc-csv", default=None, help="Specific aggregate/current AUC CSV to summarize.")
    parser.add_argument("--partition-summary", default=None, help="Specific partition_summary.csv path.")
    parser.add_argument("--max-evals", type=int, default=500, help="Fallback budget if it cannot be inferred from CSV name.")
    parser.add_argument("--boxplots", action="store_true", help="Regenerate aggregate AUC/time boxplots and summary CSVs.")
    parser.add_argument("--html-report", action="store_true", help="Regenerate xai_results_report.html.")
    parser.add_argument("--outlier-report", action="store_true", help="Create sortable HTML report for extreme AUC rows.")
    parser.add_argument("--outlier-top-n", type=int, default=50, help="Include at least this many worst rows for each AUC metric.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
