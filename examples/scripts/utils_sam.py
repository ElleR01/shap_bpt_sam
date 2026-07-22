from email.mime import image

from scipy import ndimage as ndi
import numpy as np

def cap_partition_labels(partitions, max_labels=64):
    partitions = np.asarray(partitions).astype(np.int64)

    labels, counts = np.unique(partitions, return_counts=True)

    if len(labels) <= max_labels:
        keep_labels = labels
    else:
        # Keep the largest regions, merge tiny extras into nearest kept region
        keep_labels = labels[np.argsort(counts)[-max_labels:]]

    keep_mask = np.isin(partitions, keep_labels)

    if not np.all(keep_mask):
        # For every removed pixel, copy nearest kept label
        _, nearest_idx = ndi.distance_transform_edt(~keep_mask, return_indices=True)
        partitions = partitions.copy()
        partitions[~keep_mask] = partitions[tuple(idx[~keep_mask] for idx in nearest_idx)]

    # Remap labels to contiguous 0..K-1
    unique_ids = np.unique(partitions)
    remap = {old: new for new, old in enumerate(unique_ids)}
    partitions = np.vectorize(remap.get)(partitions).astype(np.uint8)

    return partitions


def sanitize_partitions(partitions, image_shape, max_labels=63):
    p = np.asarray(partitions).astype(np.int64, copy=True)

    if p.shape != image_shape[:2]:
        raise ValueError(f"Expected partitions shape {image_shape[:2]}, got {p.shape}")

    p[p < 0] = 0

    labels = [x for x in np.unique(p) if x > 0]
    labels = sorted(labels, key=lambda x: np.sum(p == x), reverse=True)[:max_labels]

    out = np.zeros_like(p, dtype=np.int64)
    for new_label, old_label in enumerate(labels, start=1):
        out[p == old_label] = new_label

    return out



def verify_partitions(partitions, verbose_level=1):
    if verbose_level < 1:
        return
    elif verbose_level == 1:
        print(f"Unique Masks Count: {len(np.unique(partitions))}")
    elif verbose_level >= 2:
        print(f'Partitions Shape: {partitions.shape}')
        print(f'Partitions Dtype: {partitions.dtype}')
        print(f'Partitions Min: {partitions.min()}, Max: {partitions.max()}')
        print(f"Unique Masks Count: {len(np.unique(partitions))}")
        print(f'First 20 Unique Values: {np.unique(partitions)[:20]}')


import matplotlib.pyplot as plt 

from matplotlib.colors import ListedColormap, BoundaryNorm


def black_rainbow_colormap(
    n: int,
    rainbow_name: str = "turbo",
) -> tuple[ListedColormap, BoundaryNorm]:
    """
    Create a discrete colormap for integer indices 0, 1, ..., n.

    Rules:
    - index 0 is black;
    - indices 1 through n use discrete rainbow colors.

    Parameters
    ----------
    n:
        Maximum positive index. The colormap contains n + 1 colors.
    rainbow_name:
        Name of the Matplotlib colormap used for indices 1..n.

    Returns
    -------
    cmap:
        Discrete ListedColormap.
    norm:
        BoundaryNorm mapping integer values exactly to the colors.
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    rainbow = plt.colormaps[rainbow_name].resampled(n)
    rainbow_colors = rainbow(np.arange(n))

    colors = np.vstack([
        [0.0, 0.0, 0.0, 1.0],  # index 0: black
        rainbow_colors,         # indices 1..n
    ])

    cmap = ListedColormap(colors, name=f"black_{rainbow_name}_{n}")

    # Boundaries centered around integer values 0, 1, ..., n.
    boundaries = np.arange(-0.5, n + 1.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N)

    return cmap, norm

def plot_masks(masks, title=None, use_new_colormap=True, show_ratios=True, mask_types=['Sorted', 'Refined']):
    fig, axes = plt.subplots(1, len(masks), figsize=(5 * len(masks), 3.5))
    if use_new_colormap:
        # max_label = max(np.max(mask) for mask in masks)
        N = max(np.max(mask) for mask in masks)
        cmap, norm = black_rainbow_colormap(N, rainbow_name="turbo")
    else:   
        cmap = "tab20"
        norm = None
    for i, (mask_type, masks_) in enumerate(zip(mask_types, masks)):
        
        print(f'Verifying Partition: {mask_type}')
        verify_partitions(masks_, verbose_level=1)
        if show_ratios:
            bg_ratio = background_ratio(masks_, bg_label=0)
            print(f"Background Ratio for {mask_type}: {bg_ratio['bg_percent']:.2f}%")

        image = axes[i].imshow(masks_, cmap=cmap, norm=norm,aspect="auto")
        # image = ax.imshow(masks_sorted, cmap=cmap, norm=norm, aspect="auto")
        # plt.colorbar(im, fraction=0.046, pad=0.04, aspect=13.5)
        colorbar = fig.colorbar(image, ax=axes[i], ticks=np.arange(N + 1))
        colorbar.set_label("Index")
        axes[i].set_title(f'{mask_type} : {len(np.unique(masks_))} - BG: {bg_ratio["bg_percent"]:.2f}%', fontsize=14);
        axes[i].set_xticks([]); axes[i].set_yticks([]);
        print('-'*100)
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()


def plot_masks_single(mask, title=None, use_new_colormap=True, show_ratios=True, mask_types=['Sorted', 'Refined']):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    if use_new_colormap:
        # max_label = max(np.max(mask) for mask in masks)
        N = max(np.max(mask) for mask in masks)
        cmap, norm = black_rainbow_colormap(N, rainbow_name="turbo")
    else:   
        cmap = "tab20"
        norm = None
    for i, (mask_type, masks_) in enumerate(zip(mask_types, masks)):
        
        print(f'Verifying Partition: {mask_type}')
        verify_partitions(masks_, verbose_level=1)
        if show_ratios:
            bg_ratio = background_ratio(masks_, bg_label=0)
            print(f"Background Ratio for {mask_type}: {bg_ratio['bg_percent']:.2f}%")

        image = axes[i].imshow(masks_, cmap=cmap, norm=norm,aspect="auto")
        # image = ax.imshow(masks_sorted, cmap=cmap, norm=norm, aspect="auto")
        # plt.colorbar(im, fraction=0.046, pad=0.04, aspect=13.5)
        colorbar = fig.colorbar(image, ax=axes[i], ticks=np.arange(N + 1))
        colorbar.set_label("Index")
        axes[i].set_title(f'{mask_type} Masks : {len(np.unique(masks_))} - BG: {bg_ratio["bg_percent"]:.2f}%', fontsize=14);
        axes[i].set_xticks([]); axes[i].set_yticks([]);
        print('-'*100)
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()

def cap_partition_labels_local(partitions, max_labels=63):
    from scipy import ndimage as ndi

    partitions = np.asarray(partitions).astype(np.int64)
    labels, counts = np.unique(partitions, return_counts=True)

    if len(labels) <= max_labels:
        keep_labels = labels
    else:
        keep_labels = labels[np.argsort(counts)[-max_labels:]]

    keep_mask = np.isin(partitions, keep_labels)
    if not np.all(keep_mask):
        _, nearest_idx = ndi.distance_transform_edt(~keep_mask, return_indices=True)
        partitions = partitions.copy()
        partitions[~keep_mask] = partitions[tuple(idx[~keep_mask] for idx in nearest_idx)]

    unique_ids = np.unique(partitions)
    remap = {old: new for new, old in enumerate(unique_ids)}
    return np.vectorize(remap.get)(partitions).astype(np.uint8)

def sanitize_partitions_local(partitions, image_shape, max_labels=63):
    p = np.asarray(partitions).astype(np.int64, copy=True)
    if p.shape != image_shape[:2]:
        raise ValueError(f'Expected partitions shape {image_shape[:2]}, got {p.shape}')

    p[p < 0] = 0
    labels = [label for label in np.unique(p) if label > 0]
    labels = sorted(labels, key=lambda label: np.sum(p == label), reverse=True)[:max_labels]

    out = np.zeros_like(p, dtype=np.int64)
    for new_label, old_label in enumerate(labels, start=1):
        out[p == old_label] = new_label
    return out.astype(np.uint8)


from matplotlib.colors import ListedColormap, BoundaryNorm

def black_rainbow_colormap(
    n: int,
    rainbow_name: str = "turbo",
) -> tuple[ListedColormap, BoundaryNorm]:
    """
    Create a discrete colormap for integer indices 0, 1, ..., n.

    Rules:
    - index 0 is black;
    - indices 1 through n use discrete rainbow colors.

    Parameters
    ----------
    n:
        Maximum positive index. The colormap contains n + 1 colors.
    rainbow_name:
        Name of the Matplotlib colormap used for indices 1..n.

    Returns
    -------
    cmap:
        Discrete ListedColormap.
    norm:
        BoundaryNorm mapping integer values exactly to the colors.
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    rainbow = plt.colormaps[rainbow_name].resampled(n)
    rainbow_colors = rainbow(np.arange(n))

    colors = np.vstack([
        [0.0, 0.0, 0.0, 1.0],  # index 0: black
        rainbow_colors,         # indices 1..n
    ])

    cmap = ListedColormap(colors, name=f"black_{rainbow_name}_{n}")

    # Boundaries centered around integer values 0, 1, ..., n.
    boundaries = np.arange(-0.5, n + 1.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N)

    return cmap, norm




# from examples.scripts.utils_sam import black_rainbow_colormap


def plot_coalitions(image,bptrees,masks_sorted,masks_refined, save_path=None, fontsize=14, use_new_colormap=True,K=8):
    # print(save_path)
    path_results = '/'.join(save_path.split('/')[:-1])
    image_id = save_path.split('/')[-1].split('_')[0]
    # f"{path_results}/{image_id}_partition_expansion.png"
    if use_new_colormap:
        # max_label = max(np.max(mask) for mask in masks)
        N = max(np.max(masks_sorted), np.max(masks_refined))
        cmap, norm = black_rainbow_colormap(N, rainbow_name="turbo")
    else:   
        cmap = "tab20"
        norm = None
    
    
    fig, axes = plt.subplots(1,4,figsize=(10, 2))
    axes[0].imshow(image, aspect="auto")
    axes[0].set_title(f'Image: {image_id}', fontsize=fontsize)

    axx = axes[1].imshow(masks_sorted, aspect="auto")#, cmap = 'bwr')
    axes[1].set_title(f'Sorted Masks : {len(np.unique(masks_sorted))}', fontsize=fontsize)
    # colorbar = fig.colorbar(axx, ax=axes[1])
    # colorbar.set_label("Index")

    axx = axes[2].imshow(masks_sorted, cmap=cmap, norm=norm, aspect="auto")
    axes[2].set_title(f'Sorted Masks : {len(np.unique(masks_sorted))}', fontsize=fontsize)
    
    colorbar = fig.colorbar(axx, ax=axes[2], ticks=np.arange(N + 1))
    colorbar.set_label("Index")
    
    axx = axes[3].imshow(masks_refined, cmap=cmap, norm=norm, aspect="auto")
    axes[3].set_title(f'Refined Masks : {len(np.unique(masks_refined))}', fontsize=fontsize)
    colorbar = fig.colorbar(axx, ax=axes[3], ticks=np.arange(N + 1))
    colorbar.set_label("Index")
    for ax in axes.ravel():
        ax.set_xticks([]) ; ax.set_yticks([])
    plt.tight_layout()

    # print(path_results, image_id)
    plt.savefig(f'{path_results}/{image_id}_image_mask.png', dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.show()

    leaves = np.zeros(K, dtype=int)
    fig, axes = plt.subplots(len(bptrees), K, figsize=(2 * K, 1.5 * len(bptrees)), squeeze=False)
    fig.suptitle(f'BPT Partition Expansion: Full Partition with New Boundary Highlighted in Red (K=1..{K})', fontsize=fontsize + 2)
    ## Each row is one BPT construction. Full boundaries stay visible; the newly added boundary is bold red.


    # for ii, image in enumerate(images):
    for ii, (bpt_key,bptree) in enumerate(bptrees.items()):

        bptree = [tree for tree in bptrees.values()][ii]
        base_segment = shap_bpt.BaseSegment()
        # root_node = shap_bpt.AxisAlignedSegment(0, bptree.width, 0, bptree.height, base_segment)
        root_node = shap_bpt.BPT_Segment(bptree, bptree.N-1, base_segment)
        segments = [root_node]
        all_nodes = [root_node]
        
        # axes[0,ii].imshow(image)
        previous_sgm = None
        for jj in range(0,K):
            previous_sgm = make_segments(segments, image)

            # Split all current frontier nodes once. Keep the full partition visible,
            # then emphasize only the newly introduced boundary in red.
            new_segments = []
            for s in segments:
                split = s.split(s, s)
                if split is None:
                    new_segments.append(s)
                    leaves[ii] += 1
                else:
                    new_segments.extend(split)
                    all_nodes.extend(split)

            segments = new_segments

            ax = axes[ii, jj]
            img = colorize(segments, image, 0)
            img = np.clip(0.2 + img * 1.1, 0, 1)
            sgm = make_segments(segments, image)
            all_boundaries = boundary_overlay(
                sgm,
                changed_mask=None,
                color=(0, 0, 0, 0.75),
                mode='thick',
            )
            new_boundary, new_boundary_mask = new_boundary_overlay(
                previous_sgm,
                sgm,
                color=(1.0, 0.0, 0.0, 1.0),
                radius=2,
            )
            ax.imshow(img)
            ax.imshow(all_boundaries)
            ax.imshow(new_boundary)

            if ii == 0:
                ax.set_title(f'Depth\nK={jj + 1}', fontsize=fontsize)
            ax.text(
                0.02,
                0.98,
                f'new={int(new_boundary_mask.sum())}',
                transform=ax.transAxes,
                va='top',
                ha='left',
                fontsize=max(8, fontsize - 4),
                color='white',
                bbox=dict(facecolor='red', alpha=0.90, edgecolor='none', pad=1.5),
            )

        axes[ii, 0].set_ylabel(bpt_key, fontsize=fontsize)
    for ax in axes.ravel():
        ax.set_xticks([]) ; ax.set_yticks([])
    # plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05, wspace=0.05, hspace=0.05)
    if save_path:
        plt.savefig(f'{save_path}', dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.show()


# BG PIXEL RATIO
import pandas as pd

def mask_label_ratios(mask, mask_name='mask', bg_label=0):
    mask = np.asarray(mask)
    if mask.ndim == 3:
        flat = mask.reshape(-1, mask.shape[-1])
        colors, counts = np.unique(flat, axis=0, return_counts=True)
        labels = [tuple(int(v) for v in color) for color in colors]
        if isinstance(bg_label, (tuple, list, np.ndarray)):
            bg_tuple = tuple(int(v) for v in bg_label)
            is_background = np.array([label == bg_tuple for label in labels])
        else:
            is_background = np.zeros(len(labels), dtype=bool)
    else:
        labels, counts = np.unique(mask, return_counts=True)
        labels = labels.astype(int)
        is_background = labels == bg_label

    total_pixels = int(mask.shape[0] * mask.shape[1])
    return pd.DataFrame({
        'mask': mask_name,
        'label': labels,
        'pixel_count': counts.astype(int),
        'ratio': counts / total_pixels,
        'percent': 100.0 * counts / total_pixels,
        'is_background': is_background,
    }).sort_values('pixel_count', ascending=False).reset_index(drop=True)


def background_ratio(mask, bg_label=0):
    mask = np.asarray(mask)
    total_pixels = int(mask.shape[0] * mask.shape[1])
    if mask.ndim == 3:
        bg = np.asarray(bg_label)
        if bg.ndim == 0:
            raise ValueError('For RGB masks, bg_label should be an RGB tuple/list, e.g. (0, 0, 0).')
        bg_pixels = int(np.all(mask == bg, axis=-1).sum())
    else:
        bg_pixels = int((mask == bg_label).sum())
    return {
        'bg_label': bg_label,
        'bg_pixels': bg_pixels,
        'bg_ratio': bg_pixels / total_pixels,
        'bg_percent': 100.0 * bg_pixels / total_pixels,
        'fg_pixels': total_pixels - bg_pixels,
        'fg_ratio': 1.0 - (bg_pixels / total_pixels),
        'fg_percent': 100.0 * (1.0 - (bg_pixels / total_pixels)),
        'total_pixels': total_pixels,
    }


def summarize_mask_ratios(mask, mask_name='mask', bg_label=0, top_n=10):
    ratios = mask_label_ratios(mask, mask_name=mask_name, bg_label=bg_label)
    bg = background_ratio(mask, bg_label=bg_label)
    print(
        f"{mask_name}: labels={len(ratios)}, "
        f"BG(label={bg_label})={bg['bg_pixels']}/{bg['total_pixels']} "
        f"({bg['bg_percent']:.2f}%), FG={bg['fg_percent']:.2f}%"
    )
    display(ratios.head(top_n))
    return ratios, bg