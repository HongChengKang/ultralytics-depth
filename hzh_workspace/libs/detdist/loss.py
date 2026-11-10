"""detdist loss: detection losses + box-level distance Huber loss.

L = box + cls + dfl + lambda_dist * L_dist
L_dist = mean over (positive anchor AND has_dist) of Huber(sigmoid(d_pred), d_norm, delta=0.1)

Only anchors whose assigned GT box carries a valid distance (d_norm >= 0) get
distance supervision; d_norm == -1 means "distance unknown", the detection part
of the loss is untouched.

lambda_dist is read from the environment (DETDIST_LAMBDA, legacy DIST_LAMBDA; default 0.5) so the same
code can be reused for a lambda=0 detection-only baseline.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils import LOGGER
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss


class DistDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss + one distance term over all positive anchors of the one-to-many head."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.dist_lambda = float(os.environ.get("DETDIST_LAMBDA", os.environ.get("DIST_LAMBDA", "0.5")))
        self.dist_delta = float(os.environ.get("DETDIST_HUBER_DELTA", os.environ.get("DIST_HUBER_DELTA", "0.1")))
        self._stat_calls = 0  # coverage bookkeeping: how many positives land on P3
        self._stat_valid = 0.0
        self._stat_all = 0.0

    # ------------------------------------------------------------------ helpers
    def _pad_dist(self, batch: dict[str, Any], bs: int, nmax: int) -> torch.Tensor:
        """Pad per-box ``batch["dist"]`` exactly like ``preprocess`` pads targets."""
        out = torch.full((bs, nmax), -1.0, device=self.device, dtype=torch.float32)
        if nmax == 0 or "dist" not in batch:
            return out
        d = batch["dist"].to(self.device, dtype=torch.float32).view(-1)
        bidx = batch["batch_idx"].to(self.device).long().view(-1)
        if d.numel() != bidx.numel():
            raise RuntimeError(f"dist/batch_idx mismatch: {d.numel()} vs {bidx.numel()}")
        counts = torch.bincount(bidx, minlength=bs)
        offsets = torch.zeros(bs + 1, dtype=torch.long, device=self.device)
        offsets[1:] = counts.cumsum(0)
        within = torch.arange(d.numel(), device=self.device) - offsets[bidx]
        out[bidx, within] = d
        return out

    def _dist_loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, Any],
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Huber on sigmoid(d_pred) vs d_norm over all positives carrying a valid distance."""
        if "dist" not in preds:
            return preds["boxes"].sum() * 0.0
        nmax = target_gt_idx.shape[1]
        if nmax == 0:
            return preds["dist"].sum() * 0.0
        padded = self._pad_dist(batch, preds["dist"].shape[0], nmax)
        tgt = padded.gather(1, target_gt_idx.long().clamp(min=0))
        valid = fg_mask.bool() & (tgt >= 0)
        n = valid.sum()
        if int(n) == 0:
            return preds["dist"].sum() * 0.0
        self._stat_calls += 1
        self._stat_valid += float(n)
        self._stat_all += float(fg_mask.sum())
        if self._stat_calls % 50 == 0 and self._stat_all > 0:
            LOGGER.info(
                f"dist supervision: {self._stat_valid / self._stat_all * 100:.1f}% of positives carry a valid "
                f"distance over the last 50 batches"
            )
            self._stat_valid = self._stat_all = 0.0
        pred = torch.sigmoid(preds["dist"][:, 0, :])
        return F.huber_loss(pred[valid], tgt[valid], delta=self.dist_delta, reduction="sum") / n.clamp(min=1)

    # -------------------------------------------------------------------- losses
    def loss_raw(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Detection-only 3-vector (box, cls, dfl), without the batch-size factor."""
        _, loss, items = self.get_assigned_targets_and_loss(preds, batch)
        return loss, items

    def loss_with_dist(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """4-vector (box, cls, dfl, lambda*dist), without the batch-size factor."""
        (fg_mask, target_gt_idx, _tb, _ap, _st), loss, items = self.get_assigned_targets_and_loss(preds, batch)
        dist_term = self.dist_lambda * self._dist_loss(preds, batch, fg_mask, target_gt_idx)
        loss = torch.cat([loss, dist_term.view(1)])
        items = {**items, "dist_loss": dist_term.detach()}
        return loss, items

    def __call__(  # noqa: D102 - inherited signature
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        preds = self.parse_output(preds)
        loss, items = self.loss_with_dist(preds, batch)
        return loss * preds["boxes"].shape[0], items


class DistE2ELoss(E2ELoss):
    """E2ELoss with the distance term carried by the one-to-many branch."""

    def __init__(self, model: torch.nn.Module):
        super().__init__(model, loss_fn=DistDetectionLoss)

    def __call__(  # noqa: D102
        self, preds: Any, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_m4, items_m = self.one2many.loss_with_dist(one2many, batch)
        loss_o3, items_o = self.one2one.loss_raw(one2one, batch)
        total = torch.zeros_like(loss_m4)
        total[:3] = loss_m4[:3] * self.o2m + loss_o3 * self.o2o
        total[3] = loss_m4[3]  # distance term is already scaled by lambda, keep it weight-stable
        bs = one2many["boxes"].shape[0]
        items = {**items_o, "dist_loss": items_m["dist_loss"]}
        return total * bs, items
