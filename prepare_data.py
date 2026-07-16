"""
prepare_data_v2.py — Improved pseudo-label generation for belt damage detection.

Improvements over v1:
  1. Tighter bounding boxes (minAreaRect, 5px padding vs 10px)
  2. Gradient-magnitude used as辅助 (not strict AND) for edge_damage
  3. Stricter scratch thresholds (2.5/2.7/2.9 vs 2.2/2.5/2.3)
  4. Max-size filter to reject oversized boxes
  5. Better NMS with class-aware IoU=0.5 and max_per_class=8
  6. CLAHE with higher clipLimit=4.0 for better contrast
"""

import os
import shutil
import argparse
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict


DOWNSCALE = 2


def parse_yolo_polygon(label_path):
    with open(label_path) as f:
        line = f.readline().strip()
    parts = line.split()
    if len(parts) < 7:
        return None
    coords = [float(x) for x in parts[1:]]
    return [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]


def polygon_to_mask(pts, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array([(int(x * w), int(y * h)) for x, y in pts], np.int32)], 255)
    return mask


def scale_pts(pts, scale):
    return [(x * scale, y * scale) for x, y in pts]


def detect_scratches_v2(gray, belt_mask, clahe):
    h, w = gray.shape
    enhanced = clahe.apply(gray)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=belt_mask)

    local_mean = cv2.blur(enhanced.astype(np.float32), (31, 31))
    local_sq = cv2.blur(enhanced.astype(np.float32) ** 2, (31, 31))
    local_std = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0))

    belt_std_vals = local_std[belt_mask > 0]
    if len(belt_std_vals) == 0:
        return []

    dets_all = []
    for sigma_thresh in [2.5, 2.7, 2.9]:
        thresh_val = np.mean(belt_std_vals) + sigma_thresh * np.std(belt_std_vals)
        anomaly = (local_std > thresh_val).astype(np.uint8) * 255
        anomaly = cv2.bitwise_and(anomaly, anomaly, mask=belt_mask)

        for ksize in [(40, 1), (1, 40), (25, 25)]:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
            opened = cv2.morphologyEx(anomaly, cv2.MORPH_OPEN, k)
            anomaly = cv2.bitwise_or(anomaly, opened)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        anomaly = cv2.dilate(anomaly, k, iterations=3)
        anomaly = cv2.erode(anomaly, k, iterations=2)
        anomaly = cv2.bitwise_and(anomaly, anomaly, mask=belt_mask)

        contours, _ = cv2.findContours(anomaly, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 2000:
                continue
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect).astype(np.int32)
            xc, yc = box[:, 0].astype(np.float32), box[:, 1].astype(np.float32)
            x1c, y1c = max(0, int(xc.min())), max(0, int(yc.min()))
            x2c, y2c = min(w, int(xc.max())), min(h, int(yc.max()))
            bw, bh = x2c - x1c, y2c - y1c
            if bw < 20 or bh < 20:
                continue
            ar = max(bw, bh) / (min(bw, bh) + 1e-5)
            if ar < 1.8:
                continue
            if bw * bh > 0.5 * h * w:
                continue
            cx, cy = (x1c + x2c) // 2, (y1c + y2c) // 2
            if belt_mask[min(cy, h - 1), min(cx, w - 1)] == 0:
                continue
            p = 5
            dets_all.append((max(0, x1c - p), max(0, y1c - p), min(w, x2c + p), min(h, y2c + p), "scratch"))

    merged = []
    seen = set()
    for d in dets_all:
        key = (d[0] // 80, d[1] // 80, d[2] // 80, d[3] // 80)
        if key not in seen:
            seen.add(key)
            merged.append(d)
    return merged


def detect_edge_damage_v2(gray, belt_mask, clahe, poly_pts, w, h):
    enhanced = clahe.apply(gray)
    enhanced_f = enhanced.astype(np.float32)

    sobel_x = cv2.Sobel(enhanced_f, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(enhanced_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    pts_px = np.array([(int(x * w), int(y * h)) for x, y in poly_pts], np.int32)
    left_x = int(pts_px[:, 0].min())
    right_x = int(pts_px[:, 0].max())

    dets_all = []
    for band_w, dark_pct, min_area in [(60, 4, 2000), (80, 6, 1800), (100, 3, 2500)]:
        edge_band = np.zeros((h, w), dtype=np.uint8)
        edge_band[:, left_x:left_x + band_w] = 255
        edge_band[:, max(0, right_x - band_w):right_x] = 255
        edge_band = cv2.bitwise_and(edge_band, belt_mask)

        edge_vals = enhanced[edge_band > 0].astype(np.float32)
        if len(edge_vals) < 50:
            continue

        dark_thresh = np.percentile(edge_vals, dark_pct)
        dark_mask = np.zeros((h, w), dtype=np.uint8)
        dark_mask[(enhanced < dark_thresh + 12) & (edge_band > 0)] = 255

        grad_vals = grad_norm[edge_band > 0]
        grad_thresh = np.percentile(grad_vals, 80)
        grad_mask = np.zeros((h, w), dtype=np.uint8)
        grad_mask[(grad_norm > grad_thresh) & (edge_band > 0)] = 255

        dark_count = cv2.countNonZero(dark_mask)
        grad_count = cv2.countNonZero(grad_mask)

        if dark_count > grad_count * 3:
            combo = dark_mask
        else:
            combo = cv2.bitwise_or(dark_mask, grad_mask)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        combo = cv2.morphologyEx(combo, cv2.MORPH_CLOSE, k)
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        combo = cv2.morphologyEx(combo, cv2.MORPH_OPEN, k2)

        contours, _ = cv2.findContours(combo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 12 or bh < 12:
                continue
            if bw * bh > 0.5 * h * w:
                continue
            cx, cy = x + bw // 2, y + bh // 2
            if belt_mask[min(cy, h - 1), min(cx, w - 1)] == 0:
                continue
            is_left = cx < (left_x + band_w + 30)
            is_right = cx > (right_x - band_w - 30)
            if not is_left and not is_right:
                continue
            p = 5
            dets_all.append((max(0, x - p), max(0, y - p), min(w, x + bw + p), min(h, y + bh + p), "edge_damage"))

    merged = []
    seen = set()
    for d in dets_all:
        key = (d[0] // 80, d[1] // 80, d[2] // 80, d[3] // 80)
        if key not in seen:
            seen.add(key)
            merged.append(d)
    return merged


def nms_boxes_v2(dets, iou_thresh=0.5, max_per_class=8):
    by_class = defaultdict(list)
    for d in dets:
        area = (d[2] - d[0]) * (d[3] - d[1])
        by_class[d[4]].append((d[:4], area))

    result = []
    for cls, boxes in by_class.items():
        boxes.sort(key=lambda b: b[1], reverse=True)
        keep = []
        for box, area in boxes:
            ok = True
            for k, _ in keep:
                ix1 = max(box[0], k[0]); iy1 = max(box[1], k[1])
                ix2 = min(box[2], k[2]); iy2 = min(box[3], k[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                a1 = (box[2] - box[0]) * (box[3] - box[1])
                a2 = (k[2] - k[0]) * (k[3] - k[1])
                if inter / (a1 + a2 - inter + 1e-6) > iou_thresh:
                    ok = False
                    break
            if ok:
                keep.append((box, area))
        for b, _ in keep[:max_per_class]:
            result.append((b[0], b[1], b[2], b[3], cls))
    return result


def process_dataset(src_img, src_lbl, out_dir, val_ratio=0.15, seed=42):
    np.random.seed(seed)
    for s in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(out_dir, s), exist_ok=True)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    imgs = sorted(f for f in os.listdir(src_img) if f.lower().endswith((".jpg", ".jpeg", ".png")))

    stats = {"total": 0, "scratch": 0, "edge": 0, "none": 0, "ann": 0}
    records = []

    for idx, name in enumerate(imgs):
        lbl_name = os.path.splitext(name)[0] + ".txt"
        lbl_path = os.path.join(src_lbl, lbl_name)
        if not os.path.exists(lbl_path):
            continue
        img_full = cv2.imread(os.path.join(src_img, name))
        if img_full is None:
            continue
        h_full, w_full = img_full.shape[:2]

        pts_full = parse_yolo_polygon(lbl_path)
        if pts_full is None:
            continue

        ds = DOWNSCALE
        w_ds, h_ds = w_full // ds, h_full // ds
        img = cv2.resize(img_full, (w_ds, h_ds), interpolation=cv2.INTER_AREA)
        pts_ds = scale_pts(pts_full, 1.0 / ds)
        mask = polygon_to_mask(pts_ds, w_ds, h_ds)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dets_ds = detect_scratches_v2(gray, mask, clahe) + detect_edge_damage_v2(gray, mask, clahe, pts_ds, w_ds, h_ds)
        dets_ds = nms_boxes_v2(dets_ds)

        dets = []
        for x1, y1, x2, y2, cls in dets_ds:
            dets.append((x1 * ds, y1 * ds, x2 * ds, y2 * ds, cls))

        stats["total"] += 1
        stats["ann"] += len(dets)
        hs = any(d[4] == "scratch" for d in dets)
        he = any(d[4] == "edge_damage" for d in dets)
        if hs:
            stats["scratch"] += 1
        if he:
            stats["edge"] += 1
        if not hs and not he:
            stats["none"] += 1

        cmap = {"scratch": 0, "edge_damage": 1}
        yolo = []
        for x1, y1, x2, y2, cls in dets:
            yolo.append(f"{cmap[cls]} {((x1+x2)/2)/w_full:.6f} {((y1+y2)/2)/h_full:.6f} {(x2-x1)/w_full:.6f} {(y2-y1)/h_full:.6f}")
        records.append((name, yolo))

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(imgs)} images...")

    np.random.shuffle(records)
    nv = max(1, int(len(records) * val_ratio))
    val, train = records[:nv], records[nv:]

    for recs, split in [(train, "train"), (val, "val")]:
        for name, yolo in recs:
            shutil.copy2(os.path.join(src_img, name), os.path.join(out_dir, "images", split, name))
            with open(os.path.join(out_dir, "labels", split, os.path.splitext(name)[0] + ".txt"), "w") as f:
                f.write("\n".join(yolo) + "\n" if yolo else "")

    print(f"\nDataset prepared:")
    print(f"  Images: {stats['total']}  |  Annotations: {stats['ann']}  |  Avg: {stats['ann']/max(stats['total'],1):.1f}/img")
    print(f"  Scratch: {stats['scratch']} imgs  |  Edge: {stats['edge']} imgs  |  No damage: {stats['none']} imgs")
    print(f"  Train: {len(train)}  |  Val: {len(val)}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_images", type=str, default=None)
    ap.add_argument("--src_labels", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    si = os.path.abspath(args.src_images or str(root / ".." / "train" / "train" / "images"))
    sl = os.path.abspath(args.src_labels or str(root / ".." / "train" / "train" / "labels"))
    od = os.path.abspath(args.output_dir or str(root / "dataset_v2"))
    print(f"Images: {si}\nLabels: {sl}\nOutput: {od}")
    process_dataset(si, sl, od, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
