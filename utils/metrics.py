"""
metrics.py
----------
Evaluation metrics for multi-class object detection.

compute_map() - mAP@0.50 with 11-point interpolation, plus per-class AP,
                mean precision, and mean recall.
"""

import torch
from collections import defaultdict
from .loss import box_iou


def compute_map(predictions: list,
                ground_truths: list,
                num_classes: int = 4,
                iou_threshold: float = 0.50) -> dict:
    """
    Compute mAP@iou_threshold plus per-class precision / recall.

    Args:
        predictions:   list (one per image) of dicts:
                           {'boxes':  (N,4) x1y1x2y2,
                            'scores': (N,)  confidence,
                            'labels': (N,)  int class id}
        ground_truths: list (one per image) of dicts:
                           {'boxes':  (G,4),
                            'labels': (G,) int}
        num_classes:   number of damage classes (4 for RDD2022)
        iou_threshold: IoU threshold for a TP match (default 0.50)

    Returns:
        dict with keys:
            'mAP'             - float, mean over classes
            'per_class_AP'    - dict {cls_id: ap}
            'per_class_prec'  - dict {cls_id: precision}
            'per_class_rec'   - dict {cls_id: recall}
            'mean_precision'  - float
            'mean_recall'     - float
    """
    # Accumulate per-class detections (score, tp flag) and gt counts
    class_dets  = defaultdict(list)   # {cls: [(score, is_tp), ...]}
    class_n_gt  = defaultdict(int)    # {cls: total number of gt boxes}

    for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        pred_boxes  = pred['boxes']
        pred_scores = pred['scores']
        pred_labels = pred['labels']
        gt_boxes    = gt['boxes']
        gt_labels   = gt['labels']

        for cls in range(num_classes):
            p_mask = pred_labels == cls
            g_mask = gt_labels   == cls
            p_boxes  = pred_boxes[p_mask]
            p_scores = pred_scores[p_mask]
            g_boxes  = gt_boxes[g_mask]

            class_n_gt[cls] += g_boxes.shape[0]

            if p_boxes.shape[0] == 0:
                continue
            if g_boxes.shape[0] == 0:
                for s in p_scores:
                    class_dets[cls].append((s.item(), 0))
                continue

            # Greedy matching by descending confidence
            matched_gt = torch.zeros(g_boxes.shape[0], dtype=torch.bool)
            order      = p_scores.argsort(descending=True)
            for idx in order:
                ious = box_iou(p_boxes[idx].unsqueeze(0).expand(g_boxes.shape[0], 4),
                               g_boxes)   # (G,)
                best_iou, best_j = ious.max(dim=0)
                if best_iou.item() >= iou_threshold and not matched_gt[best_j.item()]:
                    matched_gt[best_j.item()] = True
                    class_dets[cls].append((p_scores[idx].item(), 1))
                else:
                    class_dets[cls].append((p_scores[idx].item(), 0))

    # Compute per-class AP with 11-point interpolation
    per_class_ap   = {}
    per_class_prec = {}
    per_class_rec  = {}

    for cls in range(num_classes):
        n_gt = class_n_gt[cls]
        dets = sorted(class_dets[cls], key=lambda x: -x[0])

        if n_gt == 0 or len(dets) == 0:
            per_class_ap[cls]   = 0.0
            per_class_prec[cls] = 0.0
            per_class_rec[cls]  = 0.0
            continue

        tp_cumsum = 0
        fp_cumsum = 0
        prec_at   = []
        rec_at    = []

        for score, is_tp in dets:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            prec = tp_cumsum / (tp_cumsum + fp_cumsum)
            rec  = tp_cumsum / n_gt
            prec_at.append(prec)
            rec_at.append(rec)

        # 11-point interpolation
        ap = 0.0
        for t in [i / 10 for i in range(11)]:
            p_interp = max((p for p, r in zip(prec_at, rec_at) if r >= t),
                           default=0.0)
            ap += p_interp / 11

        per_class_ap[cls]   = ap
        per_class_prec[cls] = prec_at[-1] if prec_at else 0.0
        per_class_rec[cls]  = rec_at[-1]  if rec_at  else 0.0

    classes_with_gt = [c for c in range(num_classes) if class_n_gt[c] > 0]
    mAP  = sum(per_class_ap[c]   for c in classes_with_gt) / max(len(classes_with_gt), 1)
    mP   = sum(per_class_prec[c] for c in classes_with_gt) / max(len(classes_with_gt), 1)
    mR   = sum(per_class_rec[c]  for c in classes_with_gt) / max(len(classes_with_gt), 1)

    return {
        'mAP':            mAP,
        'per_class_AP':   per_class_ap,
        'per_class_prec': per_class_prec,
        'per_class_rec':  per_class_rec,
        'mean_precision': mP,
        'mean_recall':    mR,
    }
