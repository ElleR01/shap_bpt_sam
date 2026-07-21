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

def plot_masks(masks, title=None, use_new_colormap=True, N = 10):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    if use_new_colormap:
        # max_label = max(np.max(mask) for mask in masks)
        
        cmap, norm = black_rainbow_colormap(N, rainbow_name="turbo")
    else:   
        cmap = "tab20"
        norm = None
    for i, (mask_type, masks_) in enumerate(zip(['Sorted', 'Refined'], masks)):
        
        print(f'Verifying Partition: {mask_type}')
        verify_partitions(masks_, verbose_level=1)
        
        image = axes[i].imshow(masks_, cmap=cmap, norm=norm,aspect="auto")
        # image = ax.imshow(masks_sorted, cmap=cmap, norm=norm, aspect="auto")
        # plt.colorbar(im, fraction=0.046, pad=0.04, aspect=13.5)
        colorbar = fig.colorbar(image, ax=axes[i], ticks=np.arange(N + 1))
        colorbar.set_label("Index")
        axes[i].set_title(f'{mask_type} Masks - Unique Count: {len(np.unique(masks_))}');
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


import numpy as np
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


# Example
N = 10
cmap, norm = black_rainbow_colormap(N)

data = np.arange(N + 1).reshape(1, -1)

fig, ax = plt.subplots(figsize=(10, 2))
image = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

ax.set_xticks(np.arange(N + 1))
ax.set_yticks([])
ax.set_title("Index 0 = black; indices 1..N = rainbow")

colorbar = fig.colorbar(image, ax=ax, ticks=np.arange(N + 1))
colorbar.set_label("Index")

plt.show()