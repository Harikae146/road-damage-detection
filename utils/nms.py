"""
nms.py
------
Non-Maximum Suppression implementations:
    standard_nms()  - classic greedy IoU threshold NMS
    soft_nms()      - Gaussian decay (Bodla et al. 2017)
    nms_per_class() - applies either method per class to avoid
                      inter-class suppression
"""

import torch
from .loss import box_iou


def standard_nms(boxes: torch.Tensor,
                 scores: torch.Tensor,
                 iou_threshold: float = 0.45) -> torch.Tensor:
    """
    Greedy NMS.
    Args:
        boxes:         (N, 4)  x1y1x2y2
        scores:        (N,)    confidence
        iou_threshold: suppress if IoU > this value
    Returns:
        keep: (K,) long tensor of surviving indices
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    order  = scores.argsort(descending=True)
    keep   = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest  = order[1:]
        ious  = box_iou(boxes[i].unsqueeze(0).expand(rest.shape[0], 4),
                        boxes[rest])
        order = rest[ious <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def soft_nms(boxes: torch.Tensor,
             scores: torch.Tensor,
             iou_threshold: float = 0.45,
             sigma: float = 0.5,
             score_threshold: float = 0.001) -> torch.Tensor:
    """
    Soft-NMS with Gaussian score decay (Bodla et al., ICCV 2017).
    Overlapping boxes are penalised rather than hard-suppressed:
        s_i <- s_i * exp(-iou^2 / sigma)

    Args:
        boxes:           (N, 4)
        scores:          (N,)
        iou_threshold:   IoU above which Gaussian decay is applied
        sigma:           Gaussian bandwidth
        score_threshold: drop boxes whose decayed score falls below this
    Returns:
        keep: (K,) indices of surviving boxes (in original order)
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    scores  = scores.clone().float()
    order   = scores.argsort(descending=True)
    boxes   = boxes.float()
    keep    = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        rest = order[1:]
        ious = box_iou(boxes[i].unsqueeze(0).expand(rest.shape[0], 4),
                       boxes[rest])

        # Gaussian decay for overlapping boxes
        decay_mask = ious > iou_threshold
        scores[rest[decay_mask]] *= torch.exp(
            -(ious[decay_mask] ** 2) / sigma)

        # Re-sort remaining by updated score and filter low-confidence
        surviving = rest[scores[rest] > score_threshold]
        order     = surviving[scores[surviving].argsort(descending=True)]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def nms_per_class(boxes: torch.Tensor,
                  scores: torch.Tensor,
                  labels: torch.Tensor,
                  iou_threshold: float = 0.45,
                  use_soft: bool = False,
                  sigma: float = 0.5) -> torch.Tensor:
    """
    Apply NMS independently per class to avoid cross-class suppression.
    Useful for road-damage detection where D00 cracks and D40 potholes
    can legitimately overlap in the same region.

    Args:
        boxes:         (N, 4)
        scores:        (N,)
        labels:        (N,)  integer class ids
        iou_threshold: passed to the selected NMS function
        use_soft:      if True use soft_nms, else standard_nms
        sigma:         Gaussian bandwidth (soft-NMS only)
    Returns:
        keep: (K,) indices into the original N boxes
    """
    fn    = soft_nms if use_soft else standard_nms
    keep  = []
    for cls_id in labels.unique():
        mask     = labels == cls_id
        idx      = mask.nonzero(as_tuple=False).squeeze(1)
        if use_soft:
            k_local = fn(boxes[idx], scores[idx],
                         iou_threshold=iou_threshold, sigma=sigma)
        else:
            k_local = fn(boxes[idx], scores[idx],
                         iou_threshold=iou_threshold)
        keep.append(idx[k_local])

    if keep:
        return torch.cat(keep)
    return torch.empty(0, dtype=torch.long, device=boxes.device)
