# Conveyor Belt Damage Detection

Detect `scratch` and `edge_damage` defects on conveyor belt images using YOLOv8.

## Setup

```bash
pip install ultralytics opencv-python-headless numpy pillow pyyaml
```

## Project Structure

```
belt_damage_detection/
├── prepare_data.py      # Generate pseudo-labels + train/val split
├── train.py             # Train YOLOv8 model
├── pipeline.py          # Inference pipeline (supports TTA)
├── evaluate.py          # Compute mF1@0.5-0.95 metric
├── data.yaml            # YOLO dataset config
├── model_weights/       # Best trained weights
├── outputs/             # Inference results (annotated images + JSONs)
└── dataset_v2/          # Prepared dataset (images + labels)
```

## Reproducing Training

```bash
# 1. Generate pseudo-labels from belt ROI annotations
python prepare_data.py

# 2. Train YOLOv8n model (100 epochs, cosine LR, strong augmentation)
python train.py --data data.yaml --model yolov8n.pt --imgsz 640 --epochs 100 --batch 8 --lr0 0.005 --patience 30 --name belt_damage_v7

# 3. Evaluate
python evaluate.py \
  --pred_dir outputs \
  --gt_dir dataset_v2/labels/train \
  --img_dir dataset_v2/images/train
```

## Inference

```bash
python pipeline.py --image_dir <path_to_images> --output_dir <output_folder>
```

Options:
- `--model <path>` — custom model weights (auto-detects from `model_weights/` or `runs/train/`)
- `--conf 0.3` — confidence threshold (default: 0.3)
- `--roi` — enable belt ROI masking
- `--tta` — enable test-time augmentation (h-flip + v-flip + NMS)

## Output Format

For each input image:
- `<name>.jpg` — original image with bounding boxes overlaid
- `<name>.json` — detections in format:
  ```json
  {"1": {"bbox_coordinates": [x_min, y_min, x_max, y_max]}, ...}
  ```

## Evaluation Results

mF1@0.5-0.95 on training set (v2 pseudo-labels):

| Metric | v6 (baseline) | v7 (improved) |
|--------|--------------|---------------|
| mF1@0.5-0.95 | 0.4864 | **0.5943** |
| F1 @ IoU=0.50 | 0.6651 | 0.7240 |
| Precision @ IoU=0.50 | 0.7128 | 0.7365 |
| Recall @ IoU=0.50 | 0.6235 | 0.7120 |

Key improvements:
- **Better pseudo-labels** (v2): gradient-magnitude + dark-region analysis for edge_damage, tighter bounding boxes (minAreaRect), stricter filtering
- **Longer training**: 100 epochs (vs 32 before early stopping), cosine LR schedule
- **Optimal confidence threshold**: 0.3 (swept 0.15-0.50)

## Model

- Architecture: YOLOv8n (nano)
- Input: 640x640
- Classes: `scratch` (0), `edge_damage` (1)
- Training: 100 epochs on v2 pseudo-labels
- Best weights: `model_weights/best.pt` (also at `runs/train/belt_damage_v7/weights/best.pt`)
