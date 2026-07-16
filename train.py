"""
train.py — Train YOLOv8 for conveyor belt damage detection.

Reproduction:
    pip install ultralytics opencv-python-headless numpy pillow pyyaml
    python prepare_data.py          # generate damage annotations + train/val split
    python train.py                 # train the model

Default hyperparameters are tuned for this conveyor belt dataset:
  - Input: 640 (balance between small-defect detection and CPU feasibility)
  - Epochs: 150 (with early stopping patience=30)
  - Batch: 8
  - Learning rate: 0.005 with cosine annealing
  - Augmentations: brightness/contrast jitter, HSV shift, CLAHE-style gamma,
    mosaic, mixup, horizontal flip — chosen for day/night lighting variation.
"""

import os
import argparse
from pathlib import Path

from ultralytics import YOLO


def get_project_root():
    return Path(__file__).resolve().parent


def train(
    data_yaml=None,
    model_size="yolov8s.pt",
    epochs=150,
    imgsz=1280,
    batch=4,
    lr0=0.005,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    close_mosaic=15,
    workers=4,
    patience=30,
    project=None,
    name="belt_damage_v6",
    device="",
    exist_ok=False,
    pretrained=True,
    optimizer="auto",
    cos_lr=True,
    resume=False,
    augment=True,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.05,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    degrees=1.0,
    translate=0.05,
    scale=0.3,
    fliplr=0.5,
    flipud=0.0,
    erasing=0.2,
    crop_fraction=1.0,
):
    """
    Train YOLOv8 model for belt damage detection.

    Augmentation rationale:
      hsv_h/s/v: color space jitter to handle night (blue/low-sat) vs day (warm/bright)
      degrees=2: slight rotation to handle camera angle variation
      mosaic=1.0: combine 4 images for small-object context
      mixup=0.15: soft blending to regularize
      fliplr=0.5: belt images are roughly symmetric horizontally
      erasing=0.3: random erasing for robustness to occlusions/timestamps
    """
    if data_yaml is None:
        data_yaml = str(get_project_root() / "data.yaml")
    if project is None:
        project = str(get_project_root() / "runs" / "train")

    if not os.path.exists(data_yaml):
        print(f"ERROR: data.yaml not found at {data_yaml}")
        print("Run 'python prepare_data.py' first to generate the dataset.")
        return None

    model = YOLO(model_size) if not resume else YOLO(os.path.join(project, name, "weights", "last.pt"))

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        warmup_momentum=warmup_momentum,
        warmup_bias_lr=warmup_bias_lr,
        close_mosaic=close_mosaic,
        workers=workers,
        patience=patience,
        project=project,
        name=name,
        device=device,
        exist_ok=exist_ok,
        pretrained=pretrained,
        optimizer=optimizer,
        cos_lr=cos_lr,
        resume=resume,
        augment=augment,
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        fliplr=fliplr,
        flipud=flipud,
        erasing=erasing,
        crop_fraction=crop_fraction,
        verbose=True,
    )

    best_weights = os.path.join(project, name, "weights", "best.pt")
    print(f"\nTraining complete. Best weights: {best_weights}")
    return best_weights


def main():
    parser = argparse.ArgumentParser(description="Train belt damage detection model")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Base model (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.005)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", type=str, default="", help="cuda device (0, cpu, etc)")
    parser.add_argument("--name", type=str, default="belt_damage_v6")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-mosaic", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        patience=args.patience,
        device=args.device,
        name=args.name,
        resume=args.resume,
        mosaic=0.0 if args.no_mosaic else 1.0,
        augment=not args.no_augment,
    )


if __name__ == "__main__":
    main()
