"""
evaluate.py
-----------
Evaluate a trained model (custom YOLOv8 or Ultralytics YOLOv8) on the
RDD2022 validation set and report mAP@0.50, precision, and recall.

Usage - custom model:
    python evaluate.py \
        --model_type custom \
        --weights ./runs/custom/best.pt \
        --model_variant standard \
        --data_dir ./dataset/rdd2022_yolo \
        --soft_nms

Usage - Ultralytics model:
    python evaluate.py \
        --model_type ultralytics \
        --weights ./runs/ultralytics/yolov8s_rdd2022/weights/best.pt \
        --data_dir ./dataset/rdd2022_yolo
"""

import argparse
import torch
from pathlib import Path

from utils.dataset import build_dataloader
from utils.metrics import compute_map
from utils.nms import nms_per_class


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type",    default="custom", choices=["custom", "ultralytics"])
    p.add_argument("--weights",       required=True,    help="Path to checkpoint (.pt)")
    p.add_argument("--model_variant", default="standard", choices=["standard", "p2"])
    p.add_argument("--data_dir",      default="./dataset/rdd2022_yolo")
    p.add_argument("--img_size",      type=int,   default=640)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--workers",       type=int,   default=4)
    p.add_argument("--conf_thresh",   type=float, default=0.25)
    p.add_argument("--iou_thresh",    type=float, default=0.45)
    p.add_argument("--soft_nms",      action="store_true")
    p.add_argument("--iou_eval",      type=float, default=0.50)
    return p.parse_args()


def evaluate_custom(args):
    from models import YOLOv8, YOLOv8P2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = YOLOv8P2 if args.model_variant == "p2" else YOLOv8
    model = model_cls(num_classes=4).to(device)

    ckpt = torch.load(args.weights, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded custom YOLOv8{'+P2' if args.model_variant=='p2' else ''} from {args.weights}")

    val_loader = build_dataloader(
        img_dir    = str(Path(args.data_dir) / "val" / "images"),
        label_dir  = str(Path(args.data_dir) / "val" / "labels"),
        img_size   = args.img_size,
        batch_size = args.batch_size,
        num_workers= args.workers,
        augment    = False,
        shuffle    = False,
    )

    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            boxes, scores, labels = model.predict(imgs, conf_thresh=args.conf_thresh)
            if boxes.shape[0] > 0:
                boxes, scores, labels = nms_per_class(
                    boxes, scores, labels,
                    use_soft  = args.soft_nms,
                    iou_thr   = args.iou_thresh,
                    score_thr = args.conf_thresh,
                )
            all_preds.append({"boxes": boxes.cpu(), "scores": scores.cpu(), "labels": labels.cpu()})
            for t in targets:
                all_targets.append({"boxes": t["boxes"], "labels": t["labels"]})

    return all_preds, all_targets


def evaluate_ultralytics(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("pip install ultralytics")

    model = YOLO(args.weights)
    data_yaml = str(Path(args.data_dir) / "rdd2022.yaml")
    metrics = model.val(
        data=data_yaml, imgsz=args.img_size, batch=args.batch_size,
        conf=args.conf_thresh, iou=args.iou_thresh,
    )
    print(f"\n=== Ultralytics YOLOv8 Evaluation ===")
    print(f"mAP@0.50  : {metrics.box.map50:.4f}")
    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")
    for i, ap in enumerate(metrics.box.ap50):
        names = ["D00_LongCrack", "D10_TransCrack", "D20_AlligCrack", "D40_Pothole"]
        print(f"  AP_{names[i]}: {ap:.4f}")
    return None, None


def main():
    args = get_args()
    if args.model_type == "ultralytics":
        evaluate_ultralytics(args)
        return

    all_preds, all_targets = evaluate_custom(args)
    results = compute_map(all_preds, all_targets, num_classes=4, iou_thr=args.iou_eval)

    nms_label = "Soft-NMS" if args.soft_nms else "Standard NMS"
    print(f"\n=== Custom YOLOv8 Evaluation ({nms_label}) ===")
    print(f"mAP@{args.iou_eval:.2f}  : {results['mAP@0.50']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    for k, v in results.items():
        if k.startswith("AP_"):
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
