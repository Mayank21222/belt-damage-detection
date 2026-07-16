"""
evaluate.py — Compute mF1@0.5–0.95 for belt damage detection.

Replicates the exact evaluation metric from the assignment:
  1. Greedy max-IoU matching between predictions and ground truth
  2. TP/FP/FN classification at each IoU threshold
  3. F1 = harmonic mean of precision and recall
  4. Average F1 across IoU thresholds 0.50 to 0.95 in steps of 0.05

Usage:
    python evaluate.py --pred_dir <folder_with_detections_json> \
                       --gt_dir <folder_with_gt_yolo_labels> \
                       --img_dir <folder_with_images> \
                       [--iou_range 0.5 0.95 0.05]

    Or run against the model directly:
    python evaluate.py --model <path_to_best.pt> \
                       --data_yaml <path_to_data.yaml> \
                       [--split val]
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def compute_iou(box_a, box_b):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / (union + 1e-10)


def greedy_match(pred_boxes, gt_boxes, iou_threshold):
    """
    Greedy matching: for each predicted box, find the GT box with max IoU.
    Spec: "Each predicted bounding box is matched to its corresponding
    ground-truth bounding box using a greedy assignment strategy based on
    maximum IoU overlap."
    Returns (tp_count, fp_count, fn_count).
    """
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0

    matched_gt = set()
    tp = 0

    for pred_idx in range(len(pred_boxes)):
        best_iou = 0
        best_gt = -1
        for gt_idx in range(len(gt_boxes)):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred_boxes[pred_idx], gt_boxes[gt_idx])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_iou >= iou_threshold and best_gt >= 0:
            tp += 1
            matched_gt.add(best_gt)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def compute_f1(tp, fp, fn):
    """Harmonic mean of precision and recall."""
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return f1


def load_yolo_labels(label_path, img_w, img_h):
    """Load YOLO labels and convert to pixel-space bounding boxes."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            boxes.append({"class": class_id, "bbox": [x1, y1, x2, y2]})
    return boxes


def load_detections_json(json_path):
    """Load detections.json and extract bounding boxes."""
    boxes = []
    if not os.path.exists(json_path):
        return boxes
    with open(json_path) as f:
        data = json.load(f)
    for key, val in data.items():
        if "bbox_coordinates" in val:
            bbox = val["bbox_coordinates"]
            cls = val.get("class", 0)
            boxes.append({"class": cls, "bbox": bbox})
    return boxes


def evaluate_detections(pred_dir, gt_dir, img_dir=None, iou_start=0.5, iou_end=0.95, iou_step=0.05):
    """Evaluate all prediction JSONs against ground truth labels."""
    thresholds = np.arange(iou_start, iou_end + iou_step / 2, iou_step)

    all_tp = {t: 0 for t in thresholds}
    all_fp = {t: 0 for t in thresholds}
    all_fn = {t: 0 for t in thresholds}

    gt_files = [f for f in os.listdir(gt_dir) if f.endswith(".txt")]
    matched_count = 0
    total_count = 0

    for gt_file in sorted(gt_files):
        base_name = os.path.splitext(gt_file)[0]
        gt_path = os.path.join(gt_dir, gt_file)
        json_path = os.path.join(pred_dir, base_name + ".json")

        img_w, img_h = 3840, 2160
        if img_dir:
            for ext in [".jpg", ".jpeg", ".png"]:
                img_path = os.path.join(img_dir, base_name + ext)
                if os.path.exists(img_path):
                    from PIL import Image
                    with Image.open(img_path) as im:
                        img_w, img_h = im.size
                    break

        gt_boxes = load_yolo_labels(gt_path, img_w, img_h)
        pred_boxes = load_detections_json(json_path)

        total_count += 1
        if pred_boxes:
            matched_count += 1

        for t in thresholds:
            tp, fp, fn = greedy_match(
                [b["bbox"] for b in pred_boxes],
                [b["bbox"] for b in gt_boxes],
                t,
            )
            all_tp[t] += tp
            all_fp[t] += fp
            all_fn[t] += fn

    per_threshold_f1 = {}
    for t in thresholds:
        f1 = compute_f1(all_tp[t], all_fp[t], all_fn[t])
        per_threshold_f1[round(float(t), 2)] = {
            "f1": round(f1, 4),
            "precision": round(all_tp[t] / (all_tp[t] + all_fp[t] + 1e-10), 4),
            "recall": round(all_tp[t] / (all_tp[t] + all_fn[t] + 1e-10), 4),
            "tp": all_tp[t],
            "fp": all_fp[t],
            "fn": all_fn[t],
        }

    mf1 = np.mean([v["f1"] for v in per_threshold_f1.values()])

    results = {
        "mF1@0.5-0.95": round(float(mf1), 4),
        "per_threshold": per_threshold_f1,
        "total_gt_boxes": sum(all_tp[t] + all_fn[t] for t in thresholds) // len(thresholds),
        "total_images": total_count,
        "images_with_predictions": matched_count,
    }

    return results


def evaluate_model(model_path, data_yaml_path, split="val"):
    """Run evaluation directly from a trained model on the dataset."""
    from ultralytics import YOLO
    from PIL import Image
    import yaml
    import tempfile
    import shutil

    with open(data_yaml_path) as f:
        data_config = yaml.safe_load(f)

    dataset_root = os.path.dirname(os.path.abspath(data_yaml_path))
    split_key = "val" if split == "val" else "train"
    img_rel = data_config.get(split_key, f"images/{split}")
    label_rel = img_rel.replace("images", "labels")

    img_dir = os.path.join(dataset_root, img_rel)
    gt_dir = os.path.join(dataset_root, label_rel)

    if not os.path.isdir(img_dir):
        print(f"ERROR: Image directory not found: {img_dir}")
        return None

    model = YOLO(model_path)

    tmp_dir = tempfile.mkdtemp()
    pred_dir = os.path.join(tmp_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    print(f"Running inference on {split} split...")
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for img_file in image_files:
        img_path = os.path.join(img_dir, img_file)
        results = model.predict(source=img_path, conf=0.25, verbose=False)

        detections = {}
        idx = 1
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                detections[str(idx)] = {
                    "bbox_coordinates": [round(x1), round(y1), round(x2), round(y2)],
                    "class": cls,
                    "confidence": round(conf, 4),
                }
                idx += 1

        base_name = os.path.splitext(img_file)[0]
        with open(os.path.join(pred_dir, base_name + ".json"), "w") as f:
            json.dump(detections, f, indent=2)

    results = evaluate_detections(pred_dir, gt_dir, img_dir)
    shutil.rmtree(tmp_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate belt damage detection (mF1@0.5-0.95)")
    parser.add_argument("--pred_dir", type=str, default=None, help="Directory with detection JSONs")
    parser.add_argument("--gt_dir", type=str, default=None, help="Directory with ground truth YOLO labels")
    parser.add_argument("--img_dir", type=str, default=None, help="Directory with images (for size info)")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model .pt")
    parser.add_argument("--data_yaml", type=str, default=None, help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "train"])
    parser.add_argument("--iou_start", type=float, default=0.5)
    parser.add_argument("--iou_end", type=float, default=0.95)
    parser.add_argument("--iou_step", type=float, default=0.05)
    args = parser.parse_args()

    if args.model and args.data_yaml:
        results = evaluate_model(args.model, args.data_yaml, args.split)
    elif args.pred_dir and args.gt_dir:
        results = evaluate_detections(
            args.pred_dir, args.gt_dir, args.img_dir, args.iou_start, args.iou_end, args.iou_step
        )
    else:
        print("Provide either (--model + --data_yaml) or (--pred_dir + --gt_dir)")
        return

    if results is None:
        return

    print(f"\n{'=' * 60}")
    print(f"  mF1@0.5-0.95: {results['mF1@0.5-0.95']:.4f}")
    print(f"  Total images: {results['total_images']}")
    print(f"  Images with predictions: {results['images_with_predictions']}")
    print(f"{'=' * 60}")
    print(f"\n{'Threshold':>10} {'F1':>8} {'Precision':>10} {'Recall':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 60)
    for t, v in results["per_threshold"].items():
        print(f"  IoU={t:.2f}  {v['f1']:>8.4f}  {v['precision']:>10.4f}  {v['recall']:>8.4f}  {v['tp']:>5}  {v['fp']:>5}  {v['fn']:>5}")

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
