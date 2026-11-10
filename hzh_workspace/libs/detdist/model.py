"""detdist model: the stock YOLO26 detection model plus a P3 distance channel.

The model YAML is unchanged (still ``Detect``); after construction the head
class is switched to :class:`detdist.head.DetectDist` and the extra 1x1 conv is
added.  Nothing else in the network differs from the reference run, so the only
delta vs. the detection baseline is the distance branch + distance loss.
"""

from __future__ import annotations

import torch

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel

from detdist.head import DetectDist
from detdist.loss import DistE2ELoss


class DistModel(DetectionModel):
    """DetectionModel whose head also regresses one normalised distance per anchor."""

    def __init__(self, cfg: str = "yolo26n.yaml", ch: int = 3, nc: int | None = None, verbose: bool = True):
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        head = self.model[-1]
        if not isinstance(head, Detect):
            raise TypeError(f"expected a Detect head as model[-1], got {type(head).__name__}")
        chs = [int(branch[0].conv.in_channels) for branch in head.cv2]
        # same object, richer behaviour: keeps every Detect attribute (stride, dfl, one2one copy)
        head.__class__ = DetectDist
        head.add_distance_branch(chs)
        head.dist.to(next(head.parameters()).device)

    def init_criterion(self):  # noqa: D102
        return DistE2ELoss(self)
