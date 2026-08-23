"""
inference.py
------------
Run inference with a trained model on a single image or a folder of images.
Detections are drawn on the image and saved to an output directory.

Usage:
    python inference.py \
        --model_type custom \
        --weights ./runs/custom/best.pt \
        --source ./sample_images/ \
        --conf_thresh 0.3 \
        --soft_nms

    python inference.py \
        --model_type ultralytics \
        --weights ./runs/ultralytics/yolov8s_rdd2022/weights/best.pt \
        --source ./sample_images/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


CLASS_NAMES  = {0: "D00_LongCrack", 1: "D10_TransCrack", 2: "D20_AlligCrack", 3: "D40_Pothole"}
CLASS_COLORS = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255), 3: (255, 165, 0)}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type",    default="custom", choices=["custom", "ultralytics"])
    p.add_argument("--weights",       required=True)
    p.add_argument("--model_variant", default="standard", choices=["standard", "p2"])
    p.add_argument("--source",        required=True, help="Image file or directory")
    p.add_argument("--out_dir",       default="./runs/inference")
    p.add_argument("--img_size",      type=int,   default=640)
    p.add_argument("--conf_thresh",   type=float, default=0.25)
    p.add_argument("--iou_thresh",    type=float, default=0.45)
    p.add_argument("--soft_nms",      action="store_true")
    return p.parse_args()


def letterbox(img: np.ndarray, target: int = 640):
    h, w = img.shape[:2]
    scale = target / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img_r = cv2.resize(img, (nw, nh))
    canvas = np.full((target, target, 3), 114, dtype=np.uint8)
    pad_top  = (target - nh) // 2
    pad_left = (target - nw) // 2
    canvas[pad_top:pad_top+nh, pad_left:pad_left+nw] = img_r
    return canvas, scale, pad_left, pad_top


def scale_boxes_back(boxes: torch.Tensor, scale: float, pad_left: int, pad_top: int):
    boxes = boxes.clone().float()
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    boxes /= scale
    return boxes


def draw_detections(img: np.ndarray, boxes, scores, labels) -> np.ndarray:
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.int().tolist()
        color = CLASS_COLORS.get(int(label), (128, 128, 128))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{CLASS_NAMES.get(int(label), '?')}: {score:.2f}"
        cv2.putText(img, text, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def collect_images(source: str):
    src = Path(source)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if src.is_file():
        return [src]
    return sorted([p for p in src.iterdir() if p.suffix.lower() in exts])


def run_custom(args):
    from models import YOLOv8, YOLOv8P2
    from utils.nms import nms_per_class

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = YOLOv8P2 if args.model_variant == "p2" else YOLOv8
    model = model_cls(num_classes=4).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in collect_images(args.source):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Cannot read {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        lb, scale, pad_l, pad_t = letterbox(img_rgb, args.img_size)
        inp = torch.from_numpy(lb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)

        with torch.no_grad():
            boxes, scores, labels = model.predict(inp, conf_thresh=args.conf_thresh)

        if boxes.shape[0] > 0:
            boxes, scores, labels = nms_per_class(
                boxes, scores, labels,
                use_soft=args.soft_nms, iou_thr=args.iou_thresh,
            )
            boxes = scale_boxes_back(boxes, scale, pad_l, pad_t)

        result = draw_detections(img_bgr.copy(), boxes, scores, labels)
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), result)
        print(f"Saved {out_path}  ({boxes.shape[0]} detections)")


def run_ultralytics(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("pip install ultralytics")

    model = YOLO(args.weights)
    model.predict(
        source=args.source, imgsz=args.img_size,
        conf=args.conf_thresh, iou=args.iou_thresh,
        save=True, project=args.out_dir, name="predict",
    )


def main():
    args = get_args()
    if args.model_type == "ultralytics":
        run_ultralytics(args)
    else:
        run_custom(args)


if __name__ == "__main__":
    main()
