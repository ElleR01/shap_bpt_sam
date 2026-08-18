# Experiments

This document describes the experiment matrix used for the full-image-set
ShapBPT/SAM evaluation on MS-COCO images.

For metric definitions and formulas, see:

```text
docs/Evaluation.md
```

## Goal

The experiments compare explanation partitions and model families under a
shared no-ground-truth evaluation protocol.

The main questions are:

- Which partition type gives better model-faithful explanations?
- How do BPT, SAM, and SAM+BPT variants compare?
- How stable are the conclusions across model families?
- How much explanation quality is gained or lost when the evaluation budget
  changes?
- How much time does each method require, including partition generation and
  explanation computation?

## Model Families

| Exp prefix | Task | Model | Output classes | Script |
| :--------- | :--- | :---- | -------------: | :----- |
| `E1` | Object detection | YOLOv11 / custom YOLO checkpoint | Usually 80 for COCO, 26 for `yolo_26` | `examples/scripts/run_yolo_full.py` |
| `E2` | Classification | ResNet-50 ImageNet | 1000 | `examples/scripts/run_resnet_full.py` |
| `E3` | Classification | ViT-B/16 ImageNet | 1000 | `examples/scripts/run_vit_full.py` |
| `E4` | Object detection | DETR ResNet-50 | COCO-style detector labels | `examples/scripts/run_detr_full.py` |

## Experiment ID Convention

The experiment ID encodes the model family and background replacement.

For non-YOLO experiments:

```text
E<model>_<background>
```

For YOLO budget ablations:

```text
E1_<background>_<budget>
```

Examples:

| Exp ID | Meaning |
| :----- | :------ |
| `E1_1_1` | YOLOv11, `noise` background, `max_evals=50` |
| `E1_1_4` | YOLOv11, `noise` background, `max_evals=500` |
| `E1_5_4` | YOLOv11, `full` background, `max_evals=500` |
| `E2_1` | ResNet-50, `noise` background |
| `E3_1` | ViT-B/16, `noise` background |
| `E4_1` | DETR, `noise` background |

## Background Replacement IDs

| ID | `--bg-type` | Meaning |
| -: | :---------- | :------ |
| `1` | `noise` | Gaussian random background, blurred with sigma=2 |
| `2` | `blurred` | Input image blurred with Gaussian sigma=8 |
| `3` | `black` | All pixels set to 0 |
| `4` | `white` | All pixels set to 255 |
| `5` | `full` | Average over black, gray, white, blurred, and noise backgrounds |
| `6` | `gray` | All pixels set to 127 |

## YOLO Budget IDs

| ID | `--max-evals` | Meaning |
| -: | ------------: | :------ |
| `1` | `50` | Very low explanation budget |
| `2` | `100` | Low explanation budget |
| `3` | `250` | Medium-low explanation budget |
| `4` | `500` | Default explanation budget |
| `5` | `1000` | High explanation budget |

Nonstandard budgets use an explicit suffix. For example:

```text
--max-evals 750 -> E1_1_custom750
```

## Explanation Methods

Each model is evaluated with the same explanation method list when the required
partition files exist.

| Method | Meaning |
| :----- | :------ |
| `BPT` | Stock BPT built from the image |
| `sam` | SAM partition labels |
| `coverage` | Coverage-adjusted SAM partitions |
| `compact` | Compact/refined partition variant |
| `filled` | Filled partition variant |
| `refined` | Refined SAM partition labels |
| `refined_filled` | Refined and filled partition labels |

For timing plots, partition time is counted as:

```text
BPT = 0 partition time
sam = time_sam
coverage = time_sam + time_coverage
compact = time_sam + time_compact
filled = time_sam + time_filler
refined = time_sam + time_refined
refined_filled = time_sam + time_refined_filled
```

The aggregate AUC plot reports `partition + explanation time`, so BPT is still
included through its explanation time.

## Saved Outputs

Each run writes one result directory, for example:

```text
examples/results/xai_results_E1_1_4
```

Common saved files:

| File | Meaning |
| :--- | :------ |
| `run_info.json` | Model, background, budget, batches, methods, paths |
| `experiment_mapping.csv` | Experiment ID mapping |
| `experiment_mapping.json` | Same mapping in JSON |
| `partition_summary.csv` | Partition generation times and counts |
| `auc_results_all_images_current_run.csv` | Incrementally updated after each completed image |
| `auc_results_all_images_<budget>_<n>.csv` | Final per-image metric table |
| `auc_results_summary_<budget>_<n>.csv` | Aggregate metric summary by method |
| `auc_results_boxplots_<budget>_<n>.png` | AUC+, AUC-, and partition+explanation time |
| `faithfulness_metric_boxplots_<budget>_<n>.png` | Aggregate no-ground-truth metric boxplots |
| `time_comparison_partition_explanation_<budget>_<n>.png` | Partition, explanation, and evaluation time |
| `total_time_by_method_<budget>_<n>.csv` | Total time table |
| `total_time_by_method_<budget>_<n>.png` | Total time plot |
| `xai_results_report.html` | Full HTML report |

Each image subdirectory contains:

| File pattern | Meaning |
| :----------- | :------ |
| `*_predictions.json` | Prediction metadata and explained class |
| `*_xai_results.png` | Input, prediction, partitions, and attribution maps |
| `auc_results_*.png` | AUC insertion/deletion curves |
| `faithfulness_curves_*.png` | Drop, insert, sufficiency, comprehensiveness, and Sensitivity-n plots |
| `<method>_<image>.png` | Single attribution map for each method |

## Core Experiment Matrix

The recommended first-pass matrix is:

| Exp ID | Model | Task | Background | Budget |
| :----- | :---- | :--- | :--------- | -----: |
| `E1_1_4` | YOLOv11 | Detection | `noise` | `500` |
| `E2_1` | ResNet-50 | Classification | `noise` | `500` |
| `E3_1` | ViT-B/16 | Classification | `noise` | `500` |
| `E4_1` | DETR | Detection | `noise` | `500` |

## YOLO Background Ablation

Run YOLO with fixed budget and different background replacements:

```text
E1_1_4, E1_2_4, E1_3_4, E1_4_4, E1_5_4
```

Command template:

```bash
for bg in noise blurred black white full; do
  python examples/scripts/run_yolo_full.py \
    --config MSCOCO_epito \
    --model yolo11s \
    --bg-type "$bg" \
    --max-evals 500 \
    --eval-batch-size 2048 \
    --auc-batch-size 1024 \
    --verbose-level medium
done
```

## YOLO Budget Ablation

Run YOLO with fixed background and different explanation budgets:

```text
E1_1_1, E1_1_2, E1_1_3, E1_1_4, E1_1_5
```

Command template:

```bash
for budget in 50 100 250 500 1000; do
  python examples/scripts/run_yolo_full.py \
    --config MSCOCO_epito \
    --model yolo11s \
    --bg-type noise \
    --max-evals "$budget" \
    --eval-batch-size 2048 \
    --auc-batch-size 1024 \
    --verbose-level medium
done
```

## Classification Baselines

Run ResNet-50:

```bash
python examples/scripts/run_resnet_full.py \
  --config MSCOCO_epito \
  --bg-type noise \
  --max-evals 500 \
  --eval-batch-size 1024 \
  --auc-batch-size 512 \
  --resnet-batch-size 1024 \
  --verbose-level medium
```

Run ViT-B/16:

```bash
python examples/scripts/run_vit_full.py \
  --config MSCOCO_epito \
  --bg-type noise \
  --max-evals 500 \
  --eval-batch-size 1024 \
  --auc-batch-size 512 \
  --vit-batch-size 1024 \
  --verbose-level medium
```

## DETR Detection Baseline

Run DETR:

```bash
python examples/scripts/run_detr_full.py \
  --config MSCOCO_epito \
  --bg-type noise \
  --max-evals 500 \
  --threshold 0.3 \
  --detr-batch-size 128 \
  --eval-batch-size 64 \
  --auc-batch-size 64 \
  --verbose-level medium
```

DETR requires a working Hugging Face `transformers` installation.

## Smoke Tests

Use `--limit 2` for a fast sanity check:

```bash
python examples/scripts/run_yolo_full.py \
  --config MSCOCO_epito \
  --limit 2 \
  --model yolo11s \
  --bg-type noise \
  --max-evals 50 \
  --verbose-level medium
```

## Summarizing Existing Results

Regenerate aggregate plots and the HTML report from saved CSVs:

```bash
python examples/scripts/summarize_results.py \
  --results-dir /path/to/results_dir \
  --boxplots \
  --html-report
```

Generate the outlier report:

```bash
python examples/scripts/summarize_results.py \
  --results-dir /path/to/results_dir \
  --outlier-report
```

Note: `summarize_results.py` can regenerate aggregate plots from CSVs, but it
cannot create per-image `faithfulness_curves_*.png` if those were not generated
during the original evaluation run.

## Recommended Paper Tables

For a paper-style report, use:

- Main table: mean and standard deviation for `auc_ins`, `auc_del`,
  `sensitivity_n_corr`, `comprehensiveness_at_20`, `sufficiency_gap_at_20`, and
  total time.
- Budget ablation table: YOLO `E1_1_1` through `E1_1_5`.
- Background ablation table: YOLO `E1_1_4` through `E1_5_4`.
- Cross-model table: `E1_1_4`, `E2_1`, `E3_1`, `E4_1`.

For formulas and metric interpretation, cite `docs/Evaluation.md`.

