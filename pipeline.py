"""
pipeline.py — Inference pipeline for conveyor belt damage detection.

Usage:
    python pipeline.py --image_dir <path_to_image_folder> --output_dir <folder>

For each input image, produces:
  1. Annotated image: <original_image_name>.jpg with bounding boxes
  2. Detections JSON: <original_image_name>.json in format:
     {
       "1": {"bbox_coordinates": [x_min, y_min, x_max, y_max]},
       "2": {"bbox_coordinates": [x_min, y_min, x_max, y_max], ...}
     }

The pipeline:
  1. Loads the belt ROI polygon from the corresponding YOLO label file
  2. Masks out non-belt regions (railings, foliage, timestamps)
  3. Runs YOLOv8 inference on the masked image
  4. Filters detections to keep only those within the belt region
  5. Outputs annotated images + JSON files
"""

import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from ultralytics import YOLO


CLASS_NAMES = {0: "scratch", 1: "edge_damage"}
CLASS_COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}


def load_belt_roi(label_path, img_w, img_h):
    """Load belt ROI polygon from YOLO label file."""
    if not os.path.exists(label_path):
        return None
    with open(label_path) as f:
        line = f.readline().strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 7:
        return None
    coords = [float(x) for x in parts[1:]]
    pts = np.array(
        [(int(coords[i] * img_w), int(coords[i + 1] * img_h)) for i in range(0, len(coords), 2)],
        dtype=np.int32,
    )
    return pts


def create_belt_mask(img_w, img_h, polygon_pts):
    """Create binary mask of the belt region."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_pts], 255)
    return mask


def mask_image(img, belt_mask):
    """Apply belt mask to image: black out non-belt regions."""
    masked = img.copy()
    masked[belt_mask == 0] = [128, 128, 128]
    return masked


def filter_detections_in_belt(detections, belt_mask):
    """Keep only detections whose center is within the belt mask."""
    h, w = belt_mask.shape
    filtered = []
    for det in detections:
        bbox = det["bbox"]
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        if belt_mask[cy, cx] > 0:
            filtered.append(det)
    return filtered


def draw_detections(img_pil, detections, font_size=20):
    """Draw bounding boxes and class labels on the image."""
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

    for det in detections:
        bbox = det["bbox"]
        cls = det.get("class", 0)
        conf = det.get("confidence", 0.0)
        color = CLASS_COLORS.get(cls, (255, 255, 0))
        label = f"{CLASS_NAMES.get(cls, f'cls{cls}')} {conf:.2f}"

        draw.rectangle(bbox, outline=color, width=3)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = bbox[0]
        text_y = max(0, bbox[1] - text_h - 4)
        draw.rectangle([text_x, text_y, text_x + text_w + 4, text_y + text_h + 4], fill=color)
        draw.text((text_x + 2, text_y + 2), label, fill=(255, 255, 255), font=font)

    return img_pil


def nms_detections(detections, iou_thresh=0.5):
    """Apply NMS across all classes."""
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    keep = []
    for d in dets:
        box = d["bbox"]
        overlap = False
        for k in keep:
            kb = k["bbox"]
            ix1 = max(box[0], kb[0]); iy1 = max(box[1], kb[1])
            ix2 = min(box[2], kb[2]); iy2 = min(box[3], kb[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            a1 = (box[2] - box[0]) * (box[3] - box[1])
            a2 = (kb[2] - kb[0]) * (kb[3] - kb[1])
            if a1 + a2 - inter > 0 and inter / (a1 + a2 - inter) > iou_thresh:
                overlap = True
                break
        if not overlap:
            keep.append(d)
    return keep


def run_pipeline(image_dir, output_dir, model_path=None, conf_threshold=0.3, use_roi=False, tta=False):
    """Run inference on all images in image_dir."""
    os.makedirs(output_dir, exist_ok=True)

    if model_path is None:
        script_dir = Path(__file__).resolve().parent
        candidates = [
            script_dir / "model_weights" / "best.pt",
            script_dir / "runs" / "train" / "belt_damage_v7" / "weights" / "best.pt",
            script_dir / "runs" / "train" / "belt_damage_v6" / "weights" / "best.pt",
            script_dir / "runs" / "detect" / "runs" / "train" / "belt_damage_v6" / "weights" / "best.pt",
            script_dir / "runs" / "detect" / "runs" / "train" / "belt_damage_v2" / "weights" / "best.pt",
            script_dir / "runs" / "train" / "belt_damage_v2" / "weights" / "best.pt",
            script_dir / "runs" / "train" / "belt_damage_v4" / "weights" / "best.pt",
            script_dir / "runs" / "train" / "belt_damage_v5" / "weights" / "best.pt",
            script_dir / "runs" / "detect" / "runs" / "train" / "belt_damage_v1" / "weights" / "best.pt",
            script_dir / "runs" / "train" / "belt_damage_v1" / "weights" / "best.pt",
        ]
        for c in candidates:
            if c.exists():
                model_path = str(c)
                break
        if model_path is None:
            print("ERROR: No model weights found. Train the model first or specify --model.")
            return

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )
    print(f"Processing {len(image_files)} images from {image_dir}" + (" (TTA)" if tta else ""))

    labels_dir_default = os.path.join(
        Path(__file__).resolve().parent, "..", "train", "train", "labels"
    )

    total_detections = 0
    images_with_detections = 0

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        base_name = os.path.splitext(img_file)[0]

        img_pil = Image.open(img_path)
        img_w, img_h = img_pil.size
        img_cv = cv2.imread(img_path)

        belt_mask = None
        if use_roi:
            label_file = base_name + ".txt"
            for search_dir in [
                os.path.join(image_dir, "..", "labels"),
                labels_dir_default,
                os.path.dirname(img_path),
            ]:
                label_path = os.path.join(search_dir, label_file)
                if os.path.exists(label_path):
                    polygon = load_belt_roi(label_path, img_w, img_h)
                    if polygon is not None:
                        belt_mask = create_belt_mask(img_w, img_h, polygon)
                    break

        if belt_mask is not None:
            masked_cv = mask_image(img_cv, belt_mask)
        else:
            masked_cv = img_cv

        all_detections = []
        results = model.predict(source=masked_cv, conf=conf_threshold, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                all_detections.append({
                    "class": cls, "confidence": conf,
                    "bbox": [round(x1), round(y1), round(x2), round(y2)],
                })

        if tta:
            flipped = cv2.flip(masked_cv, 1)
            results_flip = model.predict(source=flipped, conf=conf_threshold, verbose=False)
            for r in results_flip:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1_new = img_w - x2
                    x2_new = img_w - x1
                    cls = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    all_detections.append({
                        "class": cls, "confidence": conf,
                        "bbox": [round(x1_new), round(y1), round(x2_new), round(y2)],
                    })
            flipped_v = cv2.flip(masked_cv, -1)
            results_flip_v = model.predict(source=flipped_v, conf=conf_threshold, verbose=False)
            for r in results_flip_v:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1_new = img_w - x2
                    x2_new = img_w - x1
                    y1_new = img_h - y2
                    y2_new = img_h - y1
                    cls = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    all_detections.append({
                        "class": cls, "confidence": conf,
                        "bbox": [round(x1_new), round(y1_new), round(x2_new), round(y2_new)],
                    })

            all_detections = nms_detections(all_detections, iou_thresh=0.5)

        if belt_mask is not None:
            all_detections = filter_detections_in_belt(all_detections, belt_mask)

        all_detections.sort(key=lambda d: d["confidence"], reverse=True)

        annotated_img = draw_detections(img_pil.copy(), all_detections)
        out_img_path = os.path.join(output_dir, base_name + ".jpg")
        annotated_img.save(out_img_path, quality=95)

        det_json = {}
        for i, det in enumerate(all_detections, 1):
            det_json[str(i)] = {
                "bbox_coordinates": det["bbox"],
            }
        out_json_path = os.path.join(output_dir, base_name + ".json")
        with open(out_json_path, "w") as f:
            json.dump(det_json, f, indent=2)

        total_detections += len(all_detections)
        if all_detections:
            images_with_detections += 1

    print(f"\nInference complete:")
    print(f"  Images processed: {len(image_files)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Images with detections: {images_with_detections}")
    print(f"  Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Conveyor belt damage detection inference pipeline",
        usage="python pipeline.py --image_dir <path> --output_dir <folder>",
    )
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--roi", action="store_true", help="Enable belt ROI masking (off by default)")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation (h-flip + v-flip)")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        print(f"ERROR: Image directory not found: {args.image_dir}")
        return

    run_pipeline(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_path=args.model,
        conf_threshold=args.conf,
        use_roi=args.roi,
        tta=args.tta,
    )


if __name__ == "__main__":
    main()
