"""
neck.py
-------
PAN-FPN neck that fuses multi-scale backbone features.

Top-down path  (FPN):  P5 -> P4 -> P3 (-> P2 when use_p2=True)
Bottom-up path (PAN):  P3 -> P4 -> P5 (-> from P2 when use_p2=True)

Output channels are unified to out_ch at every level so the
detection heads receive equal-width feature maps.
"""

import torch
import torch.nn as nn
from .backbone import ConvBNSiLU, C2f


class PanFpnNeck(nn.Module):
    """
    PAN-FPN neck.  Accepts the dict produced by CSPBackbone and returns
    a list of fused tensors ordered smallest-stride first:
        3-scale:  [P3, P4, P5]          strides [8,  16, 32]
        4-scale:  [P2, P3, P4, P5]      strides [4,  8,  16, 32]
    """

    def __init__(self, backbone_channels: dict, out_ch: int = 256,
                 use_p2: bool = False, depth_mult: float = 0.33):
        """
        Args:
            backbone_channels: dict mapping 'P2'/'P3'/'P4'/'P5' -> channel count
            out_ch:  unified output channels for every level
            use_p2:  include P2 path (4-scale) for detecting tiny cracks
            depth_mult: C2f repetition multiplier (matches backbone)
        """
        super().__init__()
        self.use_p2 = use_p2
        n = max(1, round(3 * depth_mult))

        p2c = backbone_channels.get('P2', 64)
        p3c = backbone_channels['P3']
        p4c = backbone_channels['P4']
        p5c = backbone_channels['P5']

        # ── Top-down (FPN) lateral + merge convs ────────────────────────────
        self.lat_p5  = ConvBNSiLU(p5c, out_ch, 1)
        self.lat_p4  = ConvBNSiLU(p4c, out_ch, 1)
        self.lat_p3  = ConvBNSiLU(p3c, out_ch, 1)

        self.merge_p4_td = C2f(out_ch * 2, out_ch, n=n, shortcut=False)
        self.merge_p3_td = C2f(out_ch * 2, out_ch, n=n, shortcut=False)

        if use_p2:
            self.lat_p2      = ConvBNSiLU(p2c, out_ch, 1)
            self.merge_p2_td = C2f(out_ch * 2, out_ch, n=n, shortcut=False)

        # ── Bottom-up (PAN) downsampling + merge convs ───────────────────────
        self.down_p3 = ConvBNSiLU(out_ch, out_ch, 3, 2)
        self.down_p4 = ConvBNSiLU(out_ch, out_ch, 3, 2)

        self.merge_p4_bu = C2f(out_ch * 2, out_ch, n=n, shortcut=False)
        self.merge_p5_bu = C2f(out_ch * 2, out_ch, n=n, shortcut=False)

        if use_p2:
            self.down_p2     = ConvBNSiLU(out_ch, out_ch, 3, 2)
            self.merge_p3_bu = C2f(out_ch * 2, out_ch, n=n, shortcut=False)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, feats: dict):
        """
        Args:
            feats: {'P2': tensor, 'P3': tensor, 'P4': tensor, 'P5': tensor}
        Returns:
            list of tensors at each scale (smallest stride first)
        """
        p5 = self.lat_p5(feats['P5'])
        p4 = self.lat_p4(feats['P4'])
        p3 = self.lat_p3(feats['P3'])

        # Top-down
        p4_td = self.merge_p4_td(torch.cat([self.upsample(p5), p4], dim=1))
        p3_td = self.merge_p3_td(torch.cat([self.upsample(p4_td), p3], dim=1))

        if self.use_p2:
            p2 = self.lat_p2(feats['P2'])
            p2_td = self.merge_p2_td(torch.cat([self.upsample(p3_td), p2], dim=1))

        # Bottom-up
        if self.use_p2:
            p3_bu = self.merge_p3_bu(torch.cat([self.down_p2(p2_td), p3_td], dim=1))
            p4_bu = self.merge_p4_bu(torch.cat([self.down_p3(p3_bu), p4_td], dim=1))
            p5_bu = self.merge_p5_bu(torch.cat([self.down_p4(p4_bu), p5],    dim=1))
            return [p2_td, p3_bu, p4_bu, p5_bu]
        else:
            p4_bu = self.merge_p4_bu(torch.cat([self.down_p3(p3_td), p4_td], dim=1))
            p5_bu = self.merge_p5_bu(torch.cat([self.down_p4(p4_bu), p5],    dim=1))
            return [p3_td, p4_bu, p5_bu]
