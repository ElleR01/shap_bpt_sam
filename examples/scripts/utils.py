## UTIL Functions for MS COCO Dataset
from pathlib import Path
import html
import json
import os
import re

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_annotation(coco,image_no,category_name=None):
    if isinstance(image_no, str):
        image_no = int(image_no.split('\\')[-1].split('.')[0])
    
    image_info = coco.loadImgs(image_no)[0]
    if category_name is None:
        annotation_ids = coco.getAnnIds(imgIds=image_info['id'])
    else:
        category_ids = coco.getCatIds(catNms=[category_name])
        annotation_ids = coco.getAnnIds(imgIds=image_info['id'], catIds=category_ids)
    annotations = coco.loadAnns(annotation_ids)
    return annotations

def create_gt(coco,image_info,image_to_explain,image_no,category_name=None, verbose=False):
    annotations = get_annotation(coco,image_no,category_name=category_name)
    if verbose:
        if len(annotations)>0:
            print(f"Image:{image_info['id']} has {len(annotations)} annotations")
    
    mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
    category_mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

    # Combine all masks for this image
    for ann in annotations:
        if 'segmentation' in ann:
            category_id = ann['category_id']  # Unique ID for object category
            # Decode the segmentation mask
            if isinstance(ann['segmentation'], list):  # Polygon format
                for seg in ann['segmentation']:
                    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], color=1)  # Fill the mask polygon
                    cv2.fillPoly(category_mask, [pts], color=category_id)
            elif isinstance(ann['segmentation'], dict):  # RLE format
                rle = ann['segmentation']
                decoded_mask = coco.annToMask(ann)
                mask += decoded_mask  # Add binary mask
                category_mask[decoded_mask > 0] = category_id  # Assign category ID
    
    # Resize masks to match actual image dimensions
    if mask.shape[:2] != image_to_explain.shape[:2]:
        # print(f"Resizing masks: Annotated={mask.shape}, Actual={image_to_explain.shape[:2]}")
        mask = cv2.resize(mask, (image_to_explain.shape[1], image_to_explain.shape[0]), interpolation=cv2.INTER_NEAREST)
        category_mask = cv2.resize(category_mask, (image_to_explain.shape[1], image_to_explain.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask,category_mask,annotations



# Generate fake SAM-like partitions using random seed points and similarity-based assignment.
# Each pixel is assigned to the seed whose appearance+spatial descriptor is most similar.
def generate_fake_sam_partitions(image):
    # image = data["image_np"]
    H, W = image.shape[:2]
    rng = np.random.default_rng(12345)

    # Random seed points inside the image.
    num_seeds = 28
    seed_y = rng.integers(0, H, size=num_seeds)
    seed_x = rng.integers(0, W, size=num_seeds)
    seed_points = np.stack([seed_y, seed_x], axis=1)
    seed_colors = image[seed_y, seed_x].astype(np.float32)

    # Build pixel descriptors: RGB appearance + spatial coordinates.
    yy, xx = np.mgrid[0:H, 0:W]
    pixels_rgb = image.reshape(-1, 3).astype(np.float32)
    pixels_xy = np.stack([yy, xx], axis=-1).reshape(-1, 2).astype(np.float32)

    # Normalize feature scales so color and position both matter.
    color_scale = 35.0
    spatial_scale = 0.18 * max(H, W)

    # Distance from every pixel to every seed in the joint descriptor space.
    d_color = np.linalg.norm(pixels_rgb[:, None, :] - seed_colors[None, :, :], axis=-1) / color_scale
    d_space = np.linalg.norm(pixels_xy[:, None, :] - seed_points[None, :, :].astype(np.float32), axis=-1) / spatial_scale
    score = d_color + d_space
    sam_partitions = score.argmin(axis=1).reshape(H, W).astype(np.int64)

    # Optional local smoothing to make the segments more coherent.
    try:
        from scipy import ndimage as ndi
        for _ in range(2):
            sam_partitions = ndi.generic_filter(
                sam_partitions,
                function=lambda x: np.bincount(x.astype(np.int64)).argmax(),
                size=3,
                mode="nearest",
            ).astype(np.int64)
    except Exception:
        pass

    print(f"Generated fake SAM partitions with {sam_partitions.max() + 1} labels from {num_seeds} random seed points")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].imshow(image)
    ax[0].scatter(seed_x, seed_y, s=28, c="white", edgecolors="black", linewidths=0.8)
    ax[0].set_title("Random seed points")
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].imshow(image)
    ax[1].imshow(sam_partitions, cmap="tab20", alpha=0.45, interpolation="nearest")
    ax[1].set_title("Fake SAM partitions")
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    plt.tight_layout()
    plt.show()


def build_experiment_table(experiment_map):
        """Create a compact dataframe describing experiment codes and their flags."""
        experiment_table = pd.DataFrame.from_dict(experiment_map, orient="index")
        experiment_table.index.name = "Exp"
        experiment_table = experiment_table.reset_index()
        experiment_table["description"] = experiment_table.apply(
                lambda row: "+".join(
                        part for part, enabled in [
                                ("area", row["use_area_term"]),
                                ("perimeter", row["use_perim_term"]),
                                ("color", row["use_color_term"]),
                        ]
                        if enabled
                ) or "none",
                axis=1,
        )
        return experiment_table


def save_detection_summary_json(output_json, image_id, fixed_category, has_segmentation,
                                speed=None, top_k_classes=None, explained_class=None,
                                extra=None):
        """Save per-image detection metadata used by the HTML report."""
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)

        def as_builtin(value):
                if isinstance(value, np.generic):
                        return value.item()
                if isinstance(value, np.ndarray):
                        return value.tolist()
                if isinstance(value, dict):
                        return {str(k): as_builtin(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                        return [as_builtin(v) for v in value]
                return value

        summary = {
                "image_id": str(image_id),
                "fixed_category": str(fixed_category),
                "explained_class": str(explained_class if explained_class is not None else fixed_category),
                "has_segmentation": bool(has_segmentation),
                "speed": as_builtin(speed or {}),
                "top_k_classes": as_builtin(top_k_classes or []),
        }
        if extra:
                summary.update(as_builtin(extra))

        output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Detection summary saved to: {output_json}")
        return output_json, summary


def _find_detection_summary_json(results_dir, output_html, image_id, metadata_json=None):
        candidates = []
        if metadata_json is not None:
                candidates.append(Path(metadata_json))

        candidates.extend([
                Path(results_dir) / f"{image_id}_results.json",
                Path(results_dir) / f"{image_id}_detection_summary.json",
                Path(results_dir) / "detection_summary.json",
                Path(results_dir) / "results.json",
                Path(output_html).parent / f"{image_id}_results.json",
                Path(output_html).parent / f"{image_id}_detection_summary.json",
        ])

        for candidate in candidates:
                if candidate.exists():
                        return candidate
        return None


def _format_top_k_classes(top_k_classes):
        rows = []
        for item in top_k_classes or []:
                if isinstance(item, dict):
                        class_id = item.get("class_id", item.get("id", ""))
                        class_name = item.get("class_name", item.get("name", ""))
                        confidence = item.get("confidence", item.get("score", ""))
                else:
                        values = list(item)
                        class_id = values[0] if len(values) > 0 else ""
                        class_name = values[1] if len(values) > 1 else ""
                        confidence = values[2] if len(values) > 2 else ""

                if isinstance(confidence, (int, float, np.floating)):
                        confidence = f"{float(confidence):.4f}"

                rows.append(
                        f"<tr><td>{html.escape(str(class_id))}</td>"
                        f"<td>{html.escape(str(class_name))}</td>"
                        f"<td>{html.escape(str(confidence))}</td></tr>"
                )
        return "\n".join(rows)


def _path_to_img_src(path):
        if not path:
                return ""
        path = Path(path)
        if path.exists():
                return path.resolve().as_uri()
        return str(path)


def _count_mask_labels(path):
        if not path:
                return None
        path = Path(path)
        if not path.exists():
                return None

        try:
                if path.suffix.lower() == ".npy":
                        mask = np.load(path)
                else:
                        mask = plt.imread(path)
                        if mask.ndim == 3:
                                mask = mask.reshape(-1, mask.shape[-1])
                                return int(np.unique(mask, axis=0).shape[0])
                return int(np.unique(mask).size)
        except Exception:
                return None


def _infer_sam_mask_path(summary):
        if summary.get("sam_mask_path"):
                return summary["sam_mask_path"]

        image_id = summary.get("image_id_padded") or summary.get("image_id")
        if not image_id:
                return ""

        image_id = str(image_id).zfill(12)
        project_root = Path(__file__).resolve().parents[2]
        sam_path = project_root / "examples" / "partitions" / f"{image_id}_sam.png"
        return str(sam_path) if sam_path.exists() else ""


def _infer_refined_mask_path(summary):
        if summary.get("refined_mask_path"):
                return summary["refined_mask_path"]

        image_id = summary.get("image_id_padded") or summary.get("image_id")
        if not image_id:
                return ""

        image_id = str(image_id).zfill(12)
        project_root = Path(__file__).resolve().parents[2]
        refined_path = project_root / "examples" / "partitions" / f"{image_id}_refined.npy"
        return str(refined_path) if refined_path.exists() else ""


def _infer_auc_curve_path(summary, results_dir):
        if summary.get("auc_curve_path"):
                return summary["auc_curve_path"]

        image_id = summary.get("image_id")
        if not image_id:
                return ""

        auc_path = Path(results_dir) / f"auc_results_{image_id}.png"
        return str(auc_path) if auc_path.exists() else ""


def _mask_npy_to_png_src(mask_path, output_html, image_id):
        if not mask_path:
                return ""

        mask_path = Path(mask_path)
        if not mask_path.exists():
                return ""

        output_html = Path(output_html)
        preview_dir = output_html.parent / str(image_id)
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / f"{str(image_id).zfill(12)}_refined.png"

        if not preview_path.exists() or preview_path.stat().st_mtime < mask_path.stat().st_mtime:
                mask = np.load(mask_path)
                plt.imsave(preview_path, mask, cmap="tab20")

        return Path(os.path.relpath(preview_path.resolve(), start=output_html.parent.resolve())).as_posix()


def _build_detection_summary_html(summary, output_html=None, results_dir=None):
        if not summary:
                return "", "", ""

        top_k_rows = _format_top_k_classes(summary.get("top_k_classes"))
        top_k_table = ""
        if top_k_rows:
                top_k_table = f"""
                <table class="mini-table">
                    <thead><tr><th>Class ID</th><th>Class</th><th>Confidence</th></tr></thead>
                    <tbody>{top_k_rows}</tbody>
                </table>
                """

        detection_summary_html = f"""
        <div class="panel">
            <h2>Detection Summary</h2>
            <div class="summary-grid">
                <div><span class="summary-label">Image ID</span><strong>{html.escape(str(summary.get("image_id", "")))}</strong></div>
                <div><span class="summary-label">Explained class</span><strong>{html.escape(str(summary.get("explained_class", "")))}</strong></div>
                <div><span class="summary-label">Fixed category</span><strong>{html.escape(str(summary.get("fixed_category", "")))}</strong></div>
                <div><span class="summary-label">Has segmentation</span><strong>{html.escape(str(summary.get("has_segmentation", "")))}</strong></div>
            </div>
            {top_k_table}
        </div>
        """

        input_image_src = _path_to_img_src(summary.get("input_image_path"))
        sam_mask_path = _infer_sam_mask_path(summary)
        refined_mask_path = _infer_refined_mask_path(summary)
        sam_mask_src = _path_to_img_src(sam_mask_path)
        refined_mask_src = _mask_npy_to_png_src(
                refined_mask_path,
                output_html,
                summary.get("image_id", ""),
        ) if output_html is not None else ""
        sam_mask_count = _count_mask_labels(sam_mask_path)
        refined_mask_count = _count_mask_labels(refined_mask_path)
        sam_caption = f"SAM masks ({sam_mask_count})" if sam_mask_count is not None else "SAM masks"
        refined_caption = f"Refined mask ({refined_mask_count})" if refined_mask_count is not None else "Refined mask"
        input_images_html = ""
        if input_image_src or sam_mask_src or refined_mask_src:
                input_image_html = (
                        f'<figure><img src="{html.escape(input_image_src)}" alt="Input image">'
                        f'<figcaption>Input image</figcaption></figure>'
                        if input_image_src else ""
                )
                sam_mask_html = (
                        f'<figure><img src="{html.escape(sam_mask_src)}" alt="SAM masks">'
                        f'<figcaption>{html.escape(sam_caption)}</figcaption></figure>'
                        if sam_mask_src else ""
                )
                refined_mask_html = (
                        f'<figure><img src="{html.escape(refined_mask_src)}" alt="Refined mask">'
                        f'<figcaption>{html.escape(refined_caption)}</figcaption></figure>'
                        if refined_mask_src else ""
                )
                input_images_html = f"""
                <div class="panel">
                    <h2>Input, SAM Masks, And Refined Mask</h2>
                    <div class="summary-images">
                        {input_image_html}
                        {sam_mask_html}
                        {refined_mask_html}
                    </div>
                </div>
                """

        auc_curve_src = _path_to_img_src(_infer_auc_curve_path(summary, results_dir)) if results_dir is not None else ""
        auc_curve_html = ""
        if auc_curve_src:
                auc_curve_html = f"""
                <div class="panel">
                    <h2>AUC Curves</h2>
                    <figure class="auc-figure">
                        <img src="{html.escape(auc_curve_src)}" alt="AUC curves">
                    </figure>
                </div>
                """

        return detection_summary_html, input_images_html, auc_curve_html


def build_html_report(results_dir, output_html, experiment_map, experiment_table, image_id, metadata_json=None):
        """Create a standalone HTML report with filter buttons, sortable table, and figure cards."""
        results_dir = Path(results_dir)
        output_html = Path(output_html)
        image_id = str(image_id)
        output_html.parent.mkdir(parents=True, exist_ok=True)

        summary_path = _find_detection_summary_json(results_dir, output_html, image_id, metadata_json=metadata_json)
        detection_summary = None
        if summary_path is not None:
                detection_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        detection_summary_html, input_images_html, auc_curve_html = _build_detection_summary_html(
                detection_summary,
                output_html=output_html,
                results_dir=results_dir,
        )

        rows = []
        pattern = re.compile(r"^(E[1-8])_(SAM|noSAM)_(.+)\.png$")

        png_paths = sorted(results_dir.glob("E*_*.png"))
        if not png_paths:
                png_paths = sorted(results_dir.glob("*/E*_*.png"))

        for png_path in png_paths:
                match = pattern.match(png_path.name)
                if not match:
                        continue
                exp_code, sam_flag, _ = match.groups()
                cfg = experiment_map[exp_code]
                figure_path = Path(
                        os.path.relpath(png_path.resolve(), start=output_html.parent.resolve())
                ).as_posix()
                rows.append({
                        "Exp": exp_code,
                        "SAM": sam_flag,
                        "use_area_term": cfg["use_area_term"],
                        "use_perim_term": cfg["use_perim_term"],
                        "use_color_term": cfg["use_color_term"],
                        "description": experiment_table.loc[experiment_table["Exp"] == exp_code, "description"].iloc[0],
                        "figure": png_path.name,
                        "figure_path": figure_path,
                })

        df = pd.DataFrame(rows)
        if df.empty:
                print(f"No figures found in {results_dir}")
                return None
        df = df.sort_values(["Exp", "SAM"]).reset_index(drop=True)

        def make_row(r):
                return (
                        f'<tr data-exp="{r.Exp}" data-sam="{r.SAM}">'
                        f'<td>{r.Exp}</td>'
                        f'<td>{r.SAM}</td>'
                        f'<td>{"✓" if r.use_area_term else "✗"}</td>'
                        f'<td>{"✓" if r.use_perim_term else "✗"}</td>'
                        f'<td>{"✓" if r.use_color_term else "✗"}</td>'
                        f'<td>{html.escape(r.description)}</td>'
                        f'<td><a href="{html.escape(r.figure_path)}" target="_blank">{html.escape(r.figure)}</a></td>'
                        f'</tr>'
                )

        table_rows = "\n".join(df.apply(make_row, axis=1))

        card_html = []
        for _, r in df.iterrows():
                card_html.append(
                        f'''
                        <figure class="card" data-exp="{r['Exp']}" data-sam="{r['SAM']}">
                                <img src="{html.escape(r['figure_path'])}" alt="{html.escape(r['figure'])}">
                                <figcaption>
                                        <strong>{r['Exp']}</strong> · {r['SAM']}<br>
                                        {html.escape(r['description'])}
                                </figcaption>
                        </figure>
                        '''
                )

        html_text = f'''<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>ShapBPT Report - {html.escape(image_id)}</title>
    <style>
        :root {{ --panel: #111827; --muted: #94a3b8; --text: #e5e7eb; --accent: #38bdf8; }}
        body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: #0b1220; color: var(--text); }}
        .wrap {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
        h1, h2 {{ margin: 0 0 12px; }}
        .toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 14px 0 20px; }}
        .btn {{ border: 1px solid #334155; background: #111827; color: var(--text); padding: 8px 12px; border-radius: 999px; cursor: pointer; }}
        .btn.active {{ background: var(--accent); color: #00111a; border-color: var(--accent); font-weight: 700; }}
        .search {{ margin-left: auto; padding: 8px 12px; min-width: 240px; border-radius: 10px; border: 1px solid #334155; background: #0f172a; color: var(--text); }}
        .panel {{ background: rgba(17, 24, 39, 0.9); border: 1px solid #1f2937; border-radius: 18px; padding: 18px; box-shadow: 0 10px 30px rgba(0,0,0,.2); margin-bottom: 22px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px 12px; border-bottom: 1px solid #1f2937; text-align: left; vertical-align: top; }}
        th {{ position: sticky; top: 0; background: #111827; cursor: pointer; user-select: none; }}
        tr:hover td {{ background: rgba(56, 189, 248, 0.06); }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
        .card {{ margin: 0; background: #111827; border: 1px solid #1f2937; border-radius: 16px; overflow: hidden; }}
        .card img {{ width: 100%; display: block; background: white; }}
        .card figcaption {{ padding: 12px; font-size: 14px; color: #cbd5e1; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
        .summary-grid > div {{ background: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 10px 12px; }}
        .summary-wide {{ grid-column: 1 / -1; }}
        .summary-label {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
        .summary-images {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin: 12px 0 16px; }}
        .summary-images figure {{ margin: 0; background: #0f172a; border: 1px solid #334155; border-radius: 12px; overflow: hidden; }}
        .summary-images img {{ width: 100%; max-height: 360px; object-fit: contain; display: block; background: white; }}
        .summary-images figcaption {{ padding: 10px 12px; color: #cbd5e1; font-size: 14px; }}
        .auc-figure {{ margin: 0; background: #0f172a; border: 1px solid #334155; border-radius: 12px; overflow: hidden; }}
        .auc-figure img {{ width: 100%; display: block; background: white; }}
        .mini-table {{ margin-top: 14px; }}
        .collapsible-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 12px; }}
        .collapsible-head h2 {{ margin: 0; }}
        .collapse-toggle {{ border: 1px solid #334155; background: #0f172a; color: var(--text); padding: 7px 11px; border-radius: 8px; cursor: pointer; }}
        .collapse-toggle:hover {{ border-color: var(--accent); }}
        .collapsible-body.collapsed {{ display: none; }}
        .hidden {{ display: none !important; }}
        .subtle {{ color: var(--muted); }}
        a {{ color: #7dd3fc; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="wrap">
        <h1>ShapBPT report for {html.escape(image_id)}</h1>
        <div class="subtle">Filter by SAM / noSAM, click the table headers to sort, or use the search box.</div>

        {detection_summary_html}

        <div class="panel">
            <div class="collapsible-head">
                <h2>Experiment Table</h2>
                <button id="toggleExpTable" class="collapse-toggle" type="button" aria-expanded="false" aria-controls="expTablePanel">Show table</button>
            </div>
            <div id="expTablePanel" class="collapsible-body collapsed">
                <div class="toolbar">
                    <button class="btn active" data-filter="all">All</button>
                    <button class="btn" data-filter="SAM">SAM only</button>
                    <button class="btn" data-filter="noSAM">noSAM only</button>
                    {''.join(f'<button class="btn" data-filter="{exp}">{exp}</button>' for exp in sorted(df["Exp"].unique()))}
                    <input id="searchBox" class="search" type="search" placeholder="Search exp / terms / filename...">
                </div>

                <table id="expTable">
                    <thead>
                        <tr>
                            <th data-sort="string">Exp</th>
                            <th data-sort="string">SAM</th>
                            <th data-sort="string">Area</th>
                            <th data-sort="string">Perimeter</th>
                            <th data-sort="string">Color</th>
                            <th data-sort="string">Description</th>
                            <th data-sort="string">Figure</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
        </div>

        {input_images_html}

        <div class="panel">
            <h2>Figures</h2>
            <div id="figureGrid" class="grid">
                {''.join(card_html)}
            </div>
        </div>

        {auc_curve_html}
    </div>

    <script>
        const buttons = Array.from(document.querySelectorAll('.btn[data-filter]'));
        const searchBox = document.getElementById('searchBox');
        const rows = Array.from(document.querySelectorAll('#expTable tbody tr'));
        const cards = Array.from(document.querySelectorAll('#figureGrid .card'));
        const expTablePanel = document.getElementById('expTablePanel');
        const toggleExpTable = document.getElementById('toggleExpTable');
        let activeFilter = 'all';

        toggleExpTable.addEventListener('click', () => {{
            const isCollapsed = expTablePanel.classList.toggle('collapsed');
            toggleExpTable.setAttribute('aria-expanded', String(!isCollapsed));
            toggleExpTable.innerText = isCollapsed ? 'Show table' : 'Hide table';
        }});

        function applyFilters() {{
            const q = (searchBox.value || '').toLowerCase().trim();
            rows.forEach(row => {{
                const text = row.innerText.toLowerCase();
                const okFilter = activeFilter === 'all' || row.dataset.sam === activeFilter || row.dataset.exp === activeFilter;
                const okSearch = !q || text.includes(q);
                row.classList.toggle('hidden', !(okFilter && okSearch));
            }});
            cards.forEach(card => {{
                const text = card.innerText.toLowerCase();
                const okFilter = activeFilter === 'all' || card.dataset.sam === activeFilter || card.dataset.exp === activeFilter;
                const okSearch = !q || text.includes(q);
                card.classList.toggle('hidden', !(okFilter && okSearch));
            }});
        }}

        buttons.forEach(btn => btn.addEventListener('click', () => {{
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeFilter = btn.dataset.filter;
            applyFilters();
        }}));

        searchBox.addEventListener('input', applyFilters);
        applyFilters();

        const table = document.getElementById('expTable');
        const tbody = table.querySelector('tbody');
        const headers = table.querySelectorAll('th[data-sort]');
        headers.forEach((th, idx) => {{
            th.addEventListener('click', () => {{
                const asc = th.dataset.asc !== '1';
                th.dataset.asc = asc ? '1' : '0';
                const rowsArr = Array.from(tbody.querySelectorAll('tr'));
                rowsArr.sort((a, b) => {{
                    const A = a.children[idx].innerText.trim();
                    const B = b.children[idx].innerText.trim();
                    return asc ? A.localeCompare(B) : B.localeCompare(A);
                }});
                rowsArr.forEach(r => tbody.appendChild(r));
            }});
        }});
    </script>
</body>
</html>'''

        output_html.write_text(html_text, encoding='utf-8')
        print(f"HTML report saved to: {output_html}")
        return output_html, df


def build_all_images_html_report(results_root, output_html=None, title="ShapBPT All Images Report"):
        """Create one dashboard HTML with a dropdown for all per-image reports."""
        results_root = Path(results_root)
        output_html = Path(output_html) if output_html is not None else results_root / "shapbpt_all_images_report.html"
        output_html.parent.mkdir(parents=True, exist_ok=True)

        entries = []
        for json_path in sorted(results_root.glob("*/*_results.json")):
                try:
                        summary = json.loads(json_path.read_text(encoding="utf-8"))
                except Exception:
                        continue

                image_id = str(summary.get("image_id") or json_path.parent.name)
                explained_class = str(summary.get("explained_class") or summary.get("fixed_category") or "unknown")
                report_path = results_root / f"shapbpt_report_{image_id}.html"
                if not report_path.exists():
                        continue

                top = summary.get("top_k_classes") or []
                top_conf = ""
                if top:
                        first = top[0]
                        if isinstance(first, dict):
                                conf = first.get("confidence", "")
                                if isinstance(conf, (int, float, np.floating)):
                                        top_conf = f" · {float(conf):.3f}"

                entries.append({
                        "image_id": image_id,
                        "explained_class": explained_class,
                        "label": f"{image_id} - {explained_class}{top_conf}",
                        "report_path": Path(os.path.relpath(report_path.resolve(), start=output_html.parent.resolve())).as_posix(),
                        "has_segmentation": str(summary.get("has_segmentation", "")),
                })

        if not entries:
                print(f"No per-image reports found in {results_root}")
                return None

        options = "\n".join(
                f'<option value="{html.escape(entry["report_path"])}" '
                f'data-image="{html.escape(entry["image_id"])}" '
                f'data-class="{html.escape(entry["explained_class"])}" '
                f'data-seg="{html.escape(entry["has_segmentation"])}">'
                f'{html.escape(entry["label"])}</option>'
                for entry in entries
        )

        first = entries[0]
        html_text = f'''<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{html.escape(title)}</title>
    <style>
        :root {{ --panel: #111827; --muted: #94a3b8; --text: #e5e7eb; --accent: #38bdf8; }}
        body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: #0b1220; color: var(--text); }}
        .topbar {{ position: sticky; top: 0; z-index: 10; background: rgba(11, 18, 32, 0.96); border-bottom: 1px solid #1f2937; padding: 16px 24px; }}
        .wrap {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ margin: 0 0 12px; font-size: 22px; }}
        .controls {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }}
        select {{ min-width: min(560px, 100%); padding: 9px 12px; border-radius: 10px; border: 1px solid #334155; background: #0f172a; color: var(--text); }}
        .meta {{ color: var(--muted); font-size: 14px; }}
        iframe {{ width: 100%; height: calc(100vh - 118px); border: 0; display: block; background: white; }}
    </style>
</head>
<body>
    <div class="topbar">
        <div class="wrap">
            <h1>{html.escape(title)}</h1>
            <div class="controls">
                <select id="imageSelect">{options}</select>
                <span id="meta" class="meta"></span>
            </div>
        </div>
    </div>
    <iframe id="reportFrame" src="{html.escape(first["report_path"])}" title="Selected ShapBPT report"></iframe>
    <script>
        const select = document.getElementById('imageSelect');
        const frame = document.getElementById('reportFrame');
        const meta = document.getElementById('meta');

        function updateReport() {{
            const option = select.options[select.selectedIndex];
            frame.src = option.value;
            meta.textContent = `Image ${{option.dataset.image}} · explained class: ${{option.dataset.class}} · segmentation: ${{option.dataset.seg}}`;
        }}

        select.addEventListener('change', updateReport);
        updateReport();
    </script>
</body>
</html>'''

        output_html.write_text(html_text, encoding="utf-8")
        print(f"All-images HTML report saved to: {output_html}")
        return output_html, pd.DataFrame(entries)




############## EVALUATION FUNCTIONS ##############
def saliency_to_auc(nu, heatmap, f_S, f_0, predicted_cls, batch_size=4, method='del', num_samples=101, 
                    rule='trapezoid'):
    assert isinstance(heatmap, np.ndarray)
    assert len(heatmap.shape)==2 and np.issubdtype(heatmap.dtype, np.floating)

    xs, ys, ms, masks, qs = [], [], [], [], []
    for i, value in enumerate(np.linspace(start=1.0, stop=0.0, num=num_samples)):
        if method=='del':
            epsilon = (1 if value==0.0 else 0)
            q = (np.quantile(heatmap, q=value) - epsilon)
            m = heatmap <= q
            nx = (1.0 - np.sum(m) / m.size)
        elif method=='ins':
            epsilon = (1 if value==1.0 else 0)
            q = (np.quantile(heatmap, q=value) + epsilon)
            m = heatmap >= q
            nx = (np.sum(m) / m.size)
        else:
            raise Exception()
            
        # add a new datapoint on the curve
        if len(xs)==0 or nx != xs[-1]: 
            assert m.dtype==bool and len(m.shape)==2
            xs.append(nx)
            masks.append(m)
            ms.append(np.sum(heatmap[m]))
            qs.append(q)

        # evaluate the characteristic function
        if len(masks) >= batch_size or (len(masks)>0 and i==(num_samples-1)):
            y = nu(np.array(masks))[:, predicted_cls]
            ys.extend(y)
            masks = []

    assert len(masks)==0    
    xs, ys = np.array(xs), np.array(ys)
    assert(len(xs) == len(ys))

    # compute considering under/over shoots
    if f_S > f_0:
        overshoot_max = np.maximum(0, ys - f_S) # overshoot for values exceeding the maximum f(S)
        overshoot_min = np.maximum(0, f_0 - ys) # overshoot for values below the minimum f(0)
    else: # f(S) < f(0)
        overshoot_max = np.maximum(0, ys - f_0) # overshoot for values exceeding the maximum f(0)
        overshoot_min = np.maximum(0, f_S - ys) # overshoot for values below the minimum f(S)

    # clip ys, no oveshoots
    y_clipped = np.clip(ys, min(f_S, f_0), max(f_S, f_0))
    # adjust ys with the overshoot. Clip it inside the admitted range
    y_adjusted = np.clip(ys - 2*overshoot_max + 2*overshoot_min, min(f_S, f_0), max(f_S, f_0))

    # rebase to f(0)
    if f_S > f_0:
        flipped = False
        ys = ys - f_0 
        y_clipped = y_clipped - f_0 
        y_adjusted = y_adjusted - f_0
    else: # f(S) < f(0)
        flipped = True
        ys = f_0 - ys 
        y_clipped = f_0 - y_clipped 
        y_adjusted = f_0 - y_adjusted

    # rescaling
    ys_rescaled = ys / abs(f_S - f_0)
    y_clipped_rescaled = y_clipped / abs(f_S - f_0)
    y_adjusted_rescaled = y_adjusted / abs(f_S - f_0)

    auc, auc_r, auc_mae, auc_mse, auc_adj, auc_adjr, auc_clip, auc_clipr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    curve_range = range(1, len(xs)) if rule=='trapezoid' else range(len(xs))

    # compute the area under the curve with the midpoint Riemann sum (i.e. the trapezoidal rule)
    for i in curve_range:
        if rule=='trapezoid':
            delta_x = abs(xs[i] - xs[i-1])
            assert delta_x > 0
            y_mid   =         0.5*(ys[i-1] + ys[i])
            y_r_mid =         0.5*(ys_rescaled[i-1] + ys_rescaled[i])
            err_mid = y_mid - 0.5*(ms[i-1] - ms[i])
            y_clip_mid =       0.5*(y_clipped[i-1] + y_clipped[i])
            y_clipr_mid =      0.5*(y_clipped_rescaled[i-1] + y_clipped_rescaled[i])
            y_adj_mid =       0.5*(y_adjusted[i-1] + y_adjusted[i])
            y_adjr_mid =      0.5*(y_adjusted_rescaled[i-1] + y_adjusted_rescaled[i])
        else: # rectangles
            delta_x = 1.0/num_samples if i==len(xs)-1 else abs(xs[i+1] - xs[i])
            assert delta_x > 0
            y_mid   =         ys[i]
            y_r_mid =         ys_rescaled[i]
            err_mid = y_mid - ms[i]
            y_clip_mid =       y_clipped[i]
            y_clipr_mid =      y_clipped_rescaled[i]
            y_adj_mid =       y_adjusted[i]
            y_adjr_mid =      y_adjusted_rescaled[i]


        auc += abs(delta_x * y_mid) # base * height
        auc_r += abs(delta_x * y_r_mid) # base * height
        auc_mae += abs(delta_x * err_mid) # base * height
        auc_mse += abs(delta_x * (err_mid**2)) # base * height^2
        auc_clip += abs(delta_x * y_clip_mid)
        auc_clipr += abs(delta_x * y_clipr_mid)
        auc_adj += abs(delta_x * y_adj_mid)
        auc_adjr += abs(delta_x * y_adjr_mid)

    return {'xs':xs, 'ms':ms, 'qs':qs, 
            'f_0':f_0, 'f_S':f_S, 'flipped':flipped, 
            'ys':ys, 'ysr':ys_rescaled,
            'y_clip':y_clipped, 'y_clipr':y_clipped_rescaled, 
            'y_adj':y_adjusted, 'y_adjr':y_adjusted_rescaled, 
            'method':method, 'predicted_cls':predicted_cls,
            'auc':auc, 'auc_r':auc_r,
            'auc_mae':auc_mae, 'auc_mse':auc_mse, 'auc_rmse':np.sqrt(auc_mse), 
            'auc_clip':auc_clip, 'auc_clipr':auc_clipr,
            'auc_adj':auc_adj, 'auc_adjr':auc_adjr}
