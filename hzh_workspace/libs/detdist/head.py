"""Detect head with an extra box-level distance channel (detdist).

One extra 1x1 conv per detection level (P3/P4/P5) on the one-to-many branch:
dist output shape (B, 1, A_P3 + A_P4 + A_P5), concatenated in anchor order.
The prediction is squashed with sigmoid to [0, 1] and decoded as
d_m = sigmoid * 15 (15 m = DIST_MAX in the labels).

We do NOT touch the detection head structure: same cv2/cv3 towers, same
anchors, same DFL (reg_max=1 -> l1), same one2one copy for the end-to-end head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.head import Detect


class DetectDist(Detect):
    """``Detect`` + one distance channel per level on the one-to-many branch."""

    def add_distance_branch(self, chs: list[int] | tuple[int, ...]) -> None:
        """Attach one distance conv per level, initialised so sigmoid ~= 0.5 (7.5 m)."""
        self.dist = nn.ModuleList([nn.Conv2d(int(c), 1, 1) for c in chs])
        for conv in self.dist:
            nn.init.normal_(conv.weight, mean=0.0, std=0.001)
            nn.init.constant_(conv.bias, 0.0)

    def forward_head(
        self, x: list[torch.Tensor], box_head: nn.Module | None = None, cls_head: nn.Module | None = None
    ) -> dict[str, torch.Tensor]:
        """Add ``dist`` (B, 1, A_P3) to the one-to-many prediction dict."""
        out = super().forward_head(x, box_head, cls_head)
        if not out:  # fused / inference-only path
            return out
        dist = getattr(self, "dist", None)
        if dist is not None and box_head is self.one2many["box_head"]:
            bs = x[0].shape[0]
            maps = [conv(xi).view(bs, 1, -1) for xi, conv in zip(x, dist)]
            out["dist"] = torch.cat(maps, dim=2)  # anchor order: P3 -> P5
        return out

    def bias_init(self) -> None:  # pragma: no cover - called by parse_model when yaml names DetectDist
        """Keep detection bias init identical and zero the distance bias."""
        super().bias_init()
        for m in getattr(self, "dist", None) or []:
            nn.init.constant_(m.bias, 0.0)
