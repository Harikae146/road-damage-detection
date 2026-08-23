"""
loss.py
-------
YOLOv8 composite loss:
    L_total = lambda_box * L_CIoU  +  lambda_cls * L_BCE  +  lambda_dfl * L_DFL

Key components:
    ciou_loss()           - Complete IoU regression loss
    dfl_loss()            - Distribution Focal Loss for box-boundary sharpening
    task_aligned_assign() - TAL: aligns positive assignments with task quality
    YOLOv8Loss            - wraps everything, called once per training step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Loss weights (from training config / YOLOv8 paper)
LAMBDA_BOX = 7.5
LAMBDA_CLS = 0.5
LAMBDA_DFL = 1.5


# ─────────────────────────────────────────────────────────────────────────────
# Bounding-box helpers
# ─────────────────────────────────────────────────────────────────────────────

def box_iou(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """IoU between two sets of boxes (x1y1x2y2), both (N, 4)."""
    inter_x1 = torch.max(b1[:, 0], b2[:, 0])
    inter_y1 = torch.max(b1[:, 1], b2[:, 1])
    inter_x2 = torch.min(b1[:, 2], b2[:, 2])
    inter_y2 = torch.min(b1[:, 3], b2[:, 3])
    inter_w  = (inter_x2 - inter_x1).clamp(min=0)
    inter_h  = (inter_y2 - inter_y1).clamp(min=0)
    inter    = inter_w * inter_h
    area1    = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2    = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union    = area1 + area2 - inter + 1e-7
    return inter / union


def ciou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Complete IoU loss.
    pred / target: (N, 4)  in x1y1x2y2 format.
    """
    iou = box_iou(pred, target)

    # Centre distance
    pred_cx   = (pred[:, 0]   + pred[:, 2])   / 2
    pred_cy   = (pred[:, 1]   + pred[:, 3])   / 2
    tgt_cx    = (target[:, 0] + target[:, 2]) / 2
    tgt_cy    = (target[:, 1] + target[:, 3]) / 2
    rho2      = (pred_cx - tgt_cx) ** 2 + (pred_cy - tgt_cy) ** 2

    # Diagonal of enclosing box
    enc_x1 = torch.min(pred[:, 0], target[:, 0])
    enc_y1 = torch.min(pred[:, 1], target[:, 1])
    enc_x2 = torch.max(pred[:, 2], target[:, 2])
    enc_y2 = torch.max(pred[:, 3], target[:, 3])
    c2     = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7

    # Aspect-ratio consistency term
    pred_w = (pred[:, 2]   - pred[:, 0]).clamp(min=1e-7)
    pred_h = (pred[:, 3]   - pred[:, 1]).clamp(min=1e-7)
    tgt_w  = (target[:, 2] - target[:, 0]).clamp(min=1e-7)
    tgt_h  = (target[:, 3] - target[:, 1]).clamp(min=1e-7)
    v      = (4 / (torch.pi ** 2)) * (
        torch.atan(tgt_w / tgt_h) - torch.atan(pred_w / pred_h)
    ) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    return 1 - iou + rho2 / c2 + alpha * v


def dfl_loss(pred_dist: torch.Tensor, target_ltrb: torch.Tensor,
             reg_max: int = 16) -> torch.Tensor:
    """
    Distribution Focal Loss.
    pred_dist:   (N, 4 * reg_max)  raw logits
    target_ltrb: (N, 4)            continuous l/t/r/b targets (in stride units)
    """
    N = pred_dist.shape[0]
    pred_dist = pred_dist.view(N, 4, reg_max)                  # (N, 4, reg_max)
    target_ltrb = target_ltrb.clamp(0, reg_max - 1 - 1e-3)    # (N, 4)

    tl = target_ltrb.long()                   # floor bin index
    tr = tl + 1                               # ceil  bin index
    wl = tr.float() - target_ltrb            # weight for floor
    wr = target_ltrb - tl.float()            # weight for ceil

    loss = (
        F.cross_entropy(pred_dist.view(-1, reg_max),
                        tl.view(-1), reduction='none') * wl.view(-1) +
        F.cross_entropy(pred_dist.view(-1, reg_max),
                        tr.clamp(max=reg_max - 1).view(-1), reduction='none') * wr.view(-1)
    )
    return loss.view(N, 4).mean(dim=1)       # (N,)


# ─────────────────────────────────────────────────────────────────────────────
# Task-Aligned Label Assignment
# ─────────────────────────────────────────────────────────────────────────────

def task_aligned_assign(pred_boxes: torch.Tensor,
                         pred_cls:   torch.Tensor,
                         gt_boxes:   torch.Tensor,
                         gt_labels:  torch.Tensor,
                         topk: int = 10,
                         alpha: float = 0.5,
                         beta:  float = 6.0):
    """
    Task-Aligned Assigner (TAL) - assigns gt boxes to anchor points.

    Args:
        pred_boxes: (A, 4) decoded predicted boxes (x1y1x2y2)
        pred_cls:   (A, C) predicted class probabilities (sigmoid)
        gt_boxes:   (G, 4) ground-truth boxes
        gt_labels:  (G,)   class indices
        topk:       max positives per gt
        alpha/beta: alignment metric weights

    Returns:
        assigned_boxes:  (A, 4)  target boxes for each anchor
        assigned_labels: (A,)    target class index (-1 = background)
        pos_mask:        (A,)    bool, True = positive anchor
    """
    A, C = pred_cls.shape
    G    = gt_boxes.shape[0]

    if G == 0:
        return (torch.zeros(A, 4,  device=pred_boxes.device),
                torch.full((A,), -1, dtype=torch.long, device=pred_boxes.device),
                torch.zeros(A,   dtype=torch.bool,  device=pred_boxes.device))

    # Alignment metric: cls_score^alpha * iou^beta
    iou = box_iou(pred_boxes.unsqueeze(1).expand(A, G, 4).reshape(-1, 4),
                  gt_boxes.unsqueeze(0).expand(A, G, 4).reshape(-1, 4)
                  ).view(A, G)  # (A, G)

    cls_score = pred_cls[:, gt_labels]            # (A, G)
    align     = cls_score.pow(alpha) * iou.pow(beta)   # (A, G)

    # For each gt, pick topk anchors by alignment score
    topk_vals, topk_idx = align.topk(min(topk, A), dim=0)   # (topk, G)

    assigned_labels = torch.full((A,), -1, dtype=torch.long, device=pred_boxes.device)
    assigned_boxes  = torch.zeros(A, 4, device=pred_boxes.device)
    pos_mask        = torch.zeros(A, dtype=torch.bool, device=pred_boxes.device)

    for g in range(G):
        idxs = topk_idx[:, g]
        pos_mask[idxs] = True
        # Resolve conflicts: higher alignment score wins
        for i in idxs:
            i = i.item()
            g_best = align[i].argmax().item()
            assigned_labels[i] = gt_labels[g_best].item()
            assigned_boxes[i]  = gt_boxes[g_best]

    return assigned_boxes, assigned_labels, pos_mask


# ─────────────────────────────────────────────────────────────────────────────
# Full YOLOv8 Loss
# ─────────────────────────────────────────────────────────────────────────────

class YOLOv8Loss(nn.Module):
    """
    Composite YOLOv8 loss computed over all FPN scales.
    Expects raw head outputs (before decode) and a list of gt dicts.
    """
    def __init__(self, num_classes: int = 4, reg_max: int = 16,
                 strides: list = None):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides or [8, 16, 32]
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, head_outputs: list, targets: list,
                img_size: int = 640) -> torch.Tensor:
        """
        Args:
            head_outputs: list of (box_raw, cls_raw) per scale
                box_raw: (B, 4*reg_max, H, W)
                cls_raw: (B, num_classes, H, W)
            targets: list (length B) of dicts:
                {'boxes': (G,4) x1y1x2y2 pixels, 'labels': (G,) int}
            img_size: input resolution (square assumed)
        Returns:
            scalar total loss
        """
        from .head import MultiScaleHead
        device = head_outputs[0][0].device
        B = head_outputs[0][0].shape[0]

        total_box = torch.tensor(0., device=device)
        total_cls = torch.tensor(0., device=device)
        total_dfl = torch.tensor(0., device=device)
        num_pos   = 0

        for scale_idx, (box_raw, cls_raw) in enumerate(head_outputs):
            stride = self.strides[scale_idx]
            _, _, H, W = box_raw.shape

            # Decode predicted boxes
            pred_boxes = MultiScaleHead.decode_box(
                box_raw, self.reg_max, stride, device)   # (B, H*W, 4)
            pred_cls   = cls_raw.flatten(2).permute(0, 2, 1).sigmoid()   # (B, H*W, C)

            box_raw_flat = box_raw.flatten(2).permute(0, 2, 1)   # (B, H*W, 4*reg_max)

            for b in range(B):
                gt = targets[b]
                gt_boxes  = gt['boxes'].to(device)
                gt_labels = gt['labels'].to(device)

                a_boxes, a_labels, pos_mask = task_aligned_assign(
                    pred_boxes[b], pred_cls[b], gt_boxes, gt_labels)

                if pos_mask.sum() == 0:
                    continue

                num_pos += pos_mask.sum().item()

                # Box + DFL losses (positives only)
                pos_pred_box  = pred_boxes[b][pos_mask]     # (P, 4)
                pos_tgt_box   = a_boxes[pos_mask]           # (P, 4)
                pos_pred_dist = box_raw_flat[b][pos_mask]   # (P, 4*reg_max)

                # Convert target to ltrb in stride units for DFL
                cy = ((pos_tgt_box[:, 1] + pos_tgt_box[:, 3]) / 2 / stride)
                cx = ((pos_tgt_box[:, 0] + pos_tgt_box[:, 2]) / 2 / stride)
                pos_ltrb = torch.stack([
                    cx - pos_tgt_box[:, 0] / stride,
                    cy - pos_tgt_box[:, 1] / stride,
                    pos_tgt_box[:, 2] / stride - cx,
                    pos_tgt_box[:, 3] / stride - cy,
                ], dim=1)

                total_box = total_box + ciou_loss(pos_pred_box, pos_tgt_box).sum()
                total_dfl = total_dfl + dfl_loss(pos_pred_dist, pos_ltrb, self.reg_max).sum()

                # Classification loss (all anchors)
                cls_target = torch.zeros(
                    pred_cls[b].shape, device=device)
                pos_labels = a_labels[pos_mask]
                cls_target[pos_mask, pos_labels] = 1.0
                total_cls = total_cls + self.bce(
                    cls_raw.flatten(2).permute(0, 2, 1)[b],
                    cls_target).sum()

        norm = max(num_pos, 1)
        loss = (LAMBDA_BOX * total_box / norm +
                LAMBDA_CLS * total_cls / norm +
                LAMBDA_DFL * total_dfl / norm)
        return loss
