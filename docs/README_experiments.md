```bash
python examples/scripts/run_yolo_full.py --verbose-level medium


python examples/scripts/run_yolo_full.py \
  --config MSCOCO_mac \
  --limit 2 \
  --max-evals 500 \
  --verbose-level medium \
  --model yolo11s


python examples/scripts/run_resnet_full.py \
  --config MSCOCO_mac \
  --limit 2 \
  --verbose-level medium \
  --eval-batch-size 512 \
  --auc-batch-size 256 \
  --resnet-batch-size 512

python examples/scripts/summarize_results.py \
  --results-dir /path/to/resnet_xai_results \
  --boxplots \
  --html-report \
  --outlier-report

```

| Exp No |   Model   | Background Replacement Values | Details |
| :----: | :-------: | :---------------------------: | :-----: |
|  E1_1  |  yolov11  |             noise             |    -    |
|  E1_2  |  yolov11  |            blurred            |    -    |
|  E1_3  |  yolov11  |             black             |    -    |
|  E1_4  |  yolov11  |             white             |    -    |
|  E1_5  |  yolov11  |             full              |    -    |
|  E2_1  | Resnet-50 |             noise             |    -    |
