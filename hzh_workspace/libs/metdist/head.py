"""Detect head with deeper per-level distance towers, outputting log-depth.

vs detdist.head.DetectDist (v1):
  * tower = Conv3x3(ch, mid, bn+silu) + Conv2d3x3(mid, 1) per level (v1: 1x1),
    so an anchor's distance prediction sees a much larger spatial context;
  * output is log-depth in metres (no sigmoid); decode with ``logits_to_m``;
  * final bias initialised to log(5) so the untrained head reads ~5 m.

Output layout is unchanged: out["dist"] is (B, 1, A_P3+A_P4+A_P5), anchor order
P3 -> P5, so the 12-output split export / board median logic stays valid (the
export must apply exp(clamp(.)) instead of sigmoid(.)*15).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect

LOG_MIN = math.log(0.2)
LOG_MAX = math.log(15.0)


def logits_to_m(logits: torch.Tensor) -> torch.Tensor:
    """Decode log-depth logits to metres, clamped to the label range."""
    return torch.exp(logits.clamp(LOG_MIN, LOG_MAX))


class DetectDistV2(Detect):
    """``Detect`` + one deep distance tower per level on the one-to-many branch."""

    def add_distance_branch(self, chs: list[int] | tuple[int, ...], mid: int = 32, dropout: float = 0.0) -> None:
        # functional dropout in forward_head keeps state_dict keys identical to
        # dropout-free towers, so warm starts from older checkpoints still match
        self.dist_dropout = float(dropout)
        self.dist = nn.ModuleList(
            [nn.Sequential(Conv(int(c), mid, 3), nn.Conv2d(mid, 1, 3, padding=1)) for c in chs]
        )
        for tower in self.dist:
            nn.init.normal_(tower[-1].weight, mean=0.0, std=0.001)
            nn.init.constant_(tower[-1].bias, math.log(5.0))

    def forward_head(
        self, x: list[torch.Tensor], box_head: nn.Module | None = None, cls_head: nn.Module | None = None
    ) -> dict[str, torch.Tensor]:
        out = super().forward_head(x, box_head, cls_head)
        if not out:
            return out
        dist = getattr(self, "dist", None)
        if dist is not None and box_head is self.one2many["box_head"]:
            bs = x[0].shape[0]
            p = getattr(self, "dist_dropout", 0.0)
            maps = []
            for xi, tower in zip(x, dist):
                h = tower[0](xi)
                if self.training and p > 0:
                    h = torch.nn.functional.dropout2d(h, p, True)
                maps.append(tower[-1](h).view(bs, 1, -1))
            out["dist"] = torch.cat(maps, dim=2)  # log-depth logits, anchor order P3 -> P5
        return out

    def bias_init(self) -> None:  # pragma: no cover - yaml keeps Detect, runtime attachment is the path
        super().bias_init()
        for tower in getattr(self, "dist", None) or []:
            nn.init.constant_(tower[-1].bias, math.log(5.0))