# Experiment Matrix

This document tracks the full-image-set XAI experiments for MS-COCO images using
the BPT/SAM partition variants.

Each full-run script saves:

- `run_info.json`: model, background replacement, batch sizes, methods, paths.
- `experiment_mapping.csv`: experiment number mapping table.
- `experiment_mapping.json`: same mapping in JSON format.
- `auc_results_all_images_current_run.csv`: updated after each completed image.
- Aggregate AUC/time plots and an HTML report at the end of the run.

## Experiment IDs

The first part of the experiment ID identifies the model family. The suffix
identifies the background replacement value.

| Prefix | Task             | Model                            |                           Classes                           | Script                                |
| :----: | :--------------- | :------------------------------- | :---------------------------------------------------------: | :------------------------------------ |
|  `E1`  | Object detection | YOLOv11 / YOLO custom checkpoint | `len(model.names)`; usually 80 for COCO or 26 for `yolo_26` | `examples/scripts/run_yolo_full.py`   |
|  `E2`  | Classification   | ResNet-50 ImageNet               |                            1000                             | `examples/scripts/run_resnet_full.py` |
|  `E3`  | Classification   | ViT-B/16 ImageNet                |                            1000                             | `examples/scripts/run_vit_full.py`    |
|  `E4`  | Object detection | DETR ResNet-50                   |        COCO-style detection labels from model config        | `examples/scripts/run_detr_full.py`   |

## Background Replacement IDs

| Suffix | Background Replacement Values | Details                                                         |
| :----: | :---------------------------- | :-------------------------------------------------------------- |
|  `_1`  | `noise`                       | Gaussian random background, blurred with sigma=2                |
|  `_2`  | `blurred`                     | Input image blurred with Gaussian sigma=8                       |
|  `_3`  | `black`                       | All pixels set to 0                                             |
|  `_4`  | `white`                       | All pixels set to 255                                           |
|  `_5`  | `full`                        | Average over black, gray, white, blurred, and noise backgrounds |
|  `_6`  | `gray`                        | All pixels set to 127                                           |

Examples:

| Exp No | Model     | Background Replacement Values | Task           |
| :----: | :-------- | :---------------------------- | :------------- |
| `E1_1` | YOLOv11   | `noise`                       | Detection      |
| `E1_5` | YOLOv11   | `full`                        | Detection      |
| `E2_1` | ResNet-50 | `noise`                       | Classification |
| `E3_1` | ViT-B/16  | `noise`                       | Classification |
| `E4_1` | DETR      | `noise`                       | Detection      |

## Partition / Explanation Methods

All model experiments use the same explanation method set when the corresponding
partition files exist:

| Method           | Meaning                             |
| :--------------- | :---------------------------------- |
| `BPT`            | Stock BPT built from the image      |
| `sam`            | SAM partition labels                |
| `coverage`       | Coverage-adjusted SAM partitions    |
| `compact`        | Compact/refined partition variant   |
| `filled`         | Filled partition variant            |
| `refined`        | Refined SAM partition labels        |
| `refined_filled` | Refined and filled partition labels |

## E1: YOLOv11 Detection

Use this for object-detection explanations on MS-COCO. The score vector is built
from detected class confidences. Use `--class-aggregation sum` to sum detections
per class or `--class-aggregation max` to keep only the maximum confidence per
class.

Class count:

- COCO YOLO models: usually 80 classes.
- Custom `yolo_26`: 26 classes, inferred from `model.names`.

Smoke test:

```bash
python examples/scripts/run_yolo_full.py \
  --config MSCOCO_epito \
  --limit 2 \
  --max-evals 500 \
  --verbose-level medium \
  --model yolo11s \
  --bg-type noise \
  --class-aggregation sum
```

Custom YOLO-26:

```bash
python examples/scripts/run_yolo_full.py \
  --config MSCOCO_epito \
  --model yolo_26 \
  --bg-type noise \
  --class-aggregation sum
```

Default result folder for `noise`:

```text
xai_results_E1_1
```

## E2: ResNet-50 Classification

Use this for ImageNet classification explanations on the same MS-COCO images.
The score vector is the ResNet-50 softmax output over 1000 ImageNet classes.

Smoke test:

```bash
python examples/scripts/run_resnet_full.py \
  --config MSCOCO_epito \
  --limit 2 \
  --verbose-level medium \
  --bg-type noise \
  --eval-batch-size 512 \
  --auc-batch-size 256 \
  --resnet-batch-size 512
```

Default result folder for `noise`:

```text
resnet_xai_results_E2_1
```

## E3: ViT-B/16 Classification

Use this to compare a transformer classifier against ResNet-50. The score vector
is the ViT-B/16 softmax output over 1000 ImageNet classes.

Smoke test:

```bash
python examples/scripts/run_vit_full.py \
  --config MSCOCO_epito \
  --limit 2 \
  --verbose-level medium \
  --bg-type noise \
  --eval-batch-size 512 \
  --auc-batch-size 256 \
  --vit-batch-size 512
```

Default result folder for `noise`:

```text
xai_results_E3_1
```

## E4: DETR Detection

Use this to compare YOLOv11 with a transformer-based object detector. The score
vector is built from DETR post-processed detection scores per class. Like YOLO,
DETR supports:

- `--class-aggregation sum`
- `--class-aggregation max`

DETR uses Hugging Face `transformers`. Install it in the runtime environment if
needed:

```bash
pip install transformers
```

Smoke test:

```bash
python examples/scripts/run_detr_full.py \
  --config MSCOCO_mac \
  --limit 2 \
  --max-evals 50 \
  --verbose-level medium \
  --bg-type noise \
  --threshold 0.3 \
  --detr-batch-size 4 \
  --eval-batch-size 16 \
  --auc-batch-size 8
```

Default result folder for `noise`:

```text
xai_results_E4_1
```

## Running Background Ablations

Run each background replacement by changing `--bg-type`.

```bash
for bg in noise blurred black white full; do
  python examples/scripts/run_yolo_full.py \
    --config MSCOCO_epito \
    --bg-type "$bg" \
    --verbose-level medium
done
```

The scripts automatically append the experiment suffix unless disabled with:

```bash
--no-exp-suffix
```

You can also override the experiment ID manually:

```bash
--exp-no E1_1
```

## Summarizing Results

Generate boxplots, the full HTML report, and the outlier report from saved
results:

```bash
python examples/scripts/summarize_results.py \
  --results-dir /path/to/xai_results_E1_1 \
  --boxplots \
  --html-report \
  --outlier-report
```

For intermediate runs, the summarizer reads:

```text
auc_results_all_images_current_run.csv
```

This file is updated after every completed image.

## Recommended Core Matrix

At minimum, run the following for the main comparison:

| Exp No | Model     | Task           | Background |
| :----: | :-------- | :------------- | :--------- |
| `E1_1` | YOLOv11   | Detection      | `noise`    |
| `E2_1` | ResNet-50 | Classification | `noise`    |
| `E3_1` | ViT-B/16  | Classification | `noise`    |
| `E4_1` | DETR      | Detection      | `noise`    |

Then add the background ablation for YOLOv11:

```text
E1_1, E1_2, E1_3, E1_4, E1_5
```
