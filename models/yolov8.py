"""
yolov8.py
---------
Full YOLOv8 model definition wiring backbone -> neck -> head.

YOLOv8   - standard 3-scale model (P3/8, P4/16, P5/32)
YOLOv8P2 - 4-scale model adding a P2/4 head for tiny crack detection

Both expose a .predict() convenience method that runs the forward pass
and decodes raw outputs into (boxes, scores, class_ids) per image.
"""

import torch
import torch.nn as nn

from .backbone import CSPBackbone
from .neck import PanFpnNeck
from .head import MultiScaleHead


# Width & depth scaling for YOLOv8s
WIDTH_MULT  = 0.50
DEPTH_MULT  = 0.33

# Backbone output channels at YOLOv8s scale
BACKBONE_CH = {
    'P2': int(128 * WIDTH_MULT),   # 64
    'P3': int(256 * WIDTH_MULT),   # 128
    'P4': int(512 * WIDTH_MULT),   # 256
    'P5': int(512 * WIDTH_MULT),   # 256
}

# Neck unified output channels
NECK_OUT_CH = int(256 * WIDTH_MULT)   # 128

# Strides for each detection scale
STRIDES_3SCALE = [8,  16, 32]
STRIDES_4SCALE = [4,  8,  16, 32]


class YOLOv8(nn.Module):
    """
    Custom YOLOv8 (3-scale, P3/P4/P5) built from scratch.
    """
    def __init__(self, num_classes: int = 4, reg_max: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = STRIDES_3SCALE

        self.backbone = CSPBackbone(width_mult=WIDTH_MULT, depth_mult=DEPTH_MULT)
        self.neck = PanFpnNeck(BACKBONE_CH, out_ch=NECK_OUT_CH,
                               use_p2=False, depth_mult=DEPTH_MULT)
        self.head = MultiScaleHead(
            in_channels=[NECK_OUT_CH] * 3,
            num_classes=num_classes,
            reg_max=reg_max,
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W)  normalised input images
        Returns:
            list of (box_raw, cls_raw) per scale
        """
        feats  = self.backbone(x)
        fused  = self.neck(feats)
        return self.head(fused)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, conf_thresh: float = 0.25):
        """
        Run inference and decode outputs.
        Returns list (one entry per image) of dicts:
            {'boxes': (N,4), 'scores': (N,), 'labels': (N,)}
        """
        from utils.nms import nms_per_class
        from .head import MultiScaleHead as MSH

        self.eval()
        raw = self.forward(x)
        B = x.shape[0]
        device = x.device
        all_boxes, all_scores, all_labels = [], [], []

        for scale_idx, (box_raw, cls_raw) in enumerate(raw):
            stride = self.strides[scale_idx]
            boxes  = MSH.decode_box(box_raw, self.reg_max, stride, device)  # (B,HW,4)
            scores = cls_raw.flatten(2).permute(0, 2, 1).sigmoid()           # (B,HW,C)
            all_boxes.append(boxes)
            all_scores.append(scores)

        all_boxes  = torch.cat(all_boxes,  dim=1)   # (B, total_anchors, 4)
        all_scores = torch.cat(all_scores, dim=1)   # (B, total_anchors, C)

        results = []
        for b in range(B):
            s, l = all_scores[b].max(dim=-1)        # (total_anchors,)
            mask  = s > conf_thresh
            boxes_b  = all_boxes[b][mask]
            scores_b = s[mask]
            labels_b = l[mask]
            keep = nms_per_class(boxes_b, scores_b, labels_b)
            results.append({
                'boxes':  boxes_b[keep],
                'scores': scores_b[keep],
                'labels': labels_b[keep],
            })
        return results


class YOLOv8P2(YOLOv8):
    """
    YOLOv8 + P2 head (4-scale).
    Adds an extra detection level at stride 4 to detect tiny thin cracks.
    """
    def __init__(self, num_classes: int = 4, reg_max: int = 16):
        # Call nn.Module.__init__ directly to avoid YOLOv8.__init__
        nn.Module.__init__(self)
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = STRIDES_4SCALE

        self.backbone = CSPBackbone(width_mult=WIDTH_MULT, depth_mult=DEPTH_MULT)
        self.neck = PanFpnNeck(BACKBONE_CH, out_ch=NECK_OUT_CH,
                               use_p2=True, depth_mult=DEPTH_MULT)
        self.head = MultiScaleHead(
            in_channels=[NECK_OUT_CH] * 4,
            num_classes=num_classes,
            reg_max=reg_max,
        )
