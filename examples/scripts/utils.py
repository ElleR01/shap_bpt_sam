## UTIL Functions for MS COCO Dataset
from pathlib import Path
import html
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


def build_html_report(results_dir, output_html, experiment_map, experiment_table, image_id):
        """Create a standalone HTML report with filter buttons, sortable table, and figure cards."""
        results_dir = Path(results_dir)
        output_html = Path(output_html)

        rows = []
        pattern = re.compile(r"^(E[1-8])_(SAM|noSAM)_(.+)\.png$")

        for png_path in sorted(results_dir.glob("E*_*.png")):
                match = pattern.match(png_path.name)
                if not match:
                        continue
                exp_code, sam_flag, _ = match.groups()
                cfg = experiment_map[exp_code]
                rows.append({
                        "Exp": exp_code,
                        "SAM": sam_flag,
                        "use_area_term": cfg["use_area_term"],
                        "use_perim_term": cfg["use_perim_term"],
                        "use_color_term": cfg["use_color_term"],
                        "description": experiment_table.loc[experiment_table["Exp"] == exp_code, "description"].iloc[0],
                        "figure": png_path.name,
                        "figure_path": png_path.name,
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

        <div class="panel">
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

        <div class="panel">
            <h2>Figures</h2>
            <div id="figureGrid" class="grid">
                {''.join(card_html)}
            </div>
        </div>
    </div>

    <script>
        const buttons = Array.from(document.querySelectorAll('.btn[data-filter]'));
        const searchBox = document.getElementById('searchBox');
        const rows = Array.from(document.querySelectorAll('#expTable tbody tr'));
        const cards = Array.from(document.querySelectorAll('#figureGrid .card'));
        let activeFilter = 'all';

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