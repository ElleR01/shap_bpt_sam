import os
import numpy as np

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

def sam_annotations_to_label_map(annotations, image_shape):
    """Convert SAM-style annotation dicts saved in .npy files to a 2D label map."""
    annotations = list(np.asarray(annotations, dtype=object).ravel())
    if not annotations:
        return np.zeros(image_shape[:2], dtype=np.uint16)
    if not isinstance(annotations[0], dict) or 'segmentation' not in annotations[0]:
        raise ValueError(f"Expected SAM annotation dictionaries, got {type(annotations[0])}")

    h, w = image_shape[:2]
    label_map = np.zeros((h, w), dtype=np.uint16)

    # Draw large masks first, then smaller masks overwrite overlaps.
    annotations = sorted(
        annotations,
        key=lambda ann: ann.get('area', np.asarray(ann['segmentation']).sum()),
        reverse=True,
    )
    for label, ann in enumerate(annotations, start=1):
        segmentation = np.asarray(ann['segmentation'], dtype=bool)
        if segmentation.shape != (h, w):
            raise ValueError(f"Expected segmentation shape {(h, w)}, got {segmentation.shape}")
        label_map[segmentation] = label

    return label_map


def load_partition_array(path, image_shape):
    try:
        partitions = np.load(path, allow_pickle=False)
    except ValueError as exc:
        if 'Object arrays cannot be loaded' not in str(exc):
            raise
        raw = np.load(path, allow_pickle=True)
        partitions = sam_annotations_to_label_map(raw, image_shape)

    if partitions.dtype == object:
        partitions = sam_annotations_to_label_map(partitions, image_shape)

    return partitions.astype(np.uint16)


def load_refine_partitions(image_to_explain, masks_base_path, image_id, partition_type='sam', verbose=False):
    if verbose:
        print(f"Loading partitions for image_id: {image_id}, partition_type: {partition_type}")

    partition_path = os.path.join(masks_base_path, f"{image_id}_{partition_type}.npy")
    if not os.path.exists(partition_path):
        raise FileNotFoundError(f"Partition file not found: {partition_path}")

    partitions_s = load_partition_array(partition_path, image_to_explain.shape)

    # prepare_partitions(partitions, image_shape, max_labels=63)
    partition_capped_s = cap_partition_labels(partitions_s, max_labels=63)
    partition_capped_s = sanitize_partitions(partition_capped_s, image_to_explain.shape, max_labels=63)

    return partitions_s, partition_capped_s

import json

def load_json_partitions(masks_base_path, image_id, verbose=False):
    json_file_path = os.path.join(masks_base_path, f"{image_id}_results.json")
    if verbose:
        print('json_file_path:', json_file_path, os.path.exists(json_file_path))
    ## load json file if it exists, otherwise create a new dictionary
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as f:
            partition_results = json.load(f)
        if verbose:
            print('Total Time (sec):', partition_results['total_time_sec'])
            # print('Number of SAM Masks:', partitions_dict['n_sam_masks'])
            print('Number of Coverage Masks:', partition_results['n_coverage_masks'])
            print('Number of Refined Instances:', partition_results['n_refined_instances'])
            print('Number of Unique Fillers:', partition_results['n_unique_filler'])
        return partition_results
    else:
        return {}