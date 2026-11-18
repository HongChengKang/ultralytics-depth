"""detdist_v2 model: stock YOLO26 YAML + deep log-depth distance towers.

Same runtime head-class switch as v1 (yaml keeps ``Detect`` so parse_model
passes the channel tuple); the towers are attached after construction.
"""

from __future__ import annotations

import os

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel

from metdist.head import DetectDistV2
from metdist.loss import DistV2E2ELoss


class DistV2Model(DetectionModel):
    def __init__(self, cfg: str = "yolo26n.yaml", ch: int = 3, nc: int | None = None, verbose: bool = True):
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        head = self.model[-1]
        if not isinstance(head, Detect):
            raise TypeError(f"expected a Detect head as model[-1], got {type(head).__name__}")
        chs = [int(branch[0].conv.in_channels) for branch in head.cv2]
        head.__class__ = DetectDistV2
        head.add_distance_branch(
            chs,
            mid=int(os.environ.get("DETDIST_V2_MID", "32")),
            dropout=float(os.environ.get("DETDIST_V2_DROPOUT", "0")),
        )
        head.dist.to(next(head.parameters()).device)

    def init_criterion(self):  # noqa: D102
        return DistV2E2ELoss(self)