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

# --- v2 extensions: log-space asymmetric Huber + within-image order loss ---
import math
import re
from pathlib import Path

from metdist.head import LOG_MAX, LOG_MIN

OWN_RX = re.compile(r"^(video\d+(_\d+min)?_frame_\d+|camera\d+-\d+_frame_\d+)$")


class DistV2DetectionLoss(DistDetectionLoss):
    """v8DetectionLoss + log-space Huber distance term (one-to-many positives)."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.dist_lambda = float(os.environ.get("DETDIST_V2_LAMBDA", "3.0"))
        self.dist_delta = float(os.environ.get("DETDIST_V2_HUBER_DELTA", "0.2"))
        # quantile-Huber: tau<0.5 penalises overestimation more -> readings sit
        # at/below GT. Applied only to targets < dist_near_max metres; farther
        # targets stay symmetric. tau=0.5 reproduces the plain Huber behaviour.
        self.dist_tau = float(os.environ.get("DETDIST_V2_TAU", "0.5"))
        self.dist_near_weight = float(os.environ.get("DETDIST_V2_NEAR_WEIGHT", "1.0"))
        self.dist_near_max = float(os.environ.get("DETDIST_V2_NEAR_MAX", "5.0"))
        # within-image ordinal supervision (scale-invariant): lets coco/fpv
        # contribute scene diversity without injecting their absolute scale
        self.dist_rank = float(os.environ.get("DETDIST_V2_RANK", "0"))
        # with unmasked labels (labels_dist_cal): coco/fpv boxes contribute to
        # the order loss only, never to the metric regression (scale unreliable)
        self.metric_own = os.environ.get("DETDIST_V2_METRIC_OWN", "0") == "1"

    def _dist_loss(self, preds, batch, fg_mask, target_gt_idx):
        if "dist" not in preds:
            return preds["boxes"].sum() * 0.0
        nmax = target_gt_idx.shape[1]
        if nmax == 0:
            return preds["dist"].sum() * 0.0
        padded = self._pad_dist(batch, preds["dist"].shape[0], nmax)
        tgt = padded.gather(1, target_gt_idx.long().clamp(min=0))
        valid = fg_mask.bool() & (tgt >= 0)
        if self.metric_own and "im_file" in batch:
            own = torch.tensor(
                [bool(OWN_RX.match(Path(f).stem)) for f in batch["im_file"]], device=valid.device
            )
            valid = valid & own[:, None]
        n = valid.sum()
        if int(n) == 0:
            return preds["dist"].sum() * 0.0
        self._stat_calls += 1
        self._stat_valid += float(n)
        self._stat_all += float(fg_mask.sum())
        if self._stat_calls % 50 == 0 and self._stat_all > 0:
            from ultralytics.utils import LOGGER

            LOGGER.info(
                f"dist supervision: {self._stat_valid / self._stat_all * 100:.1f}% of positives carry a valid "
                f"distance over the last 50 batches"
            )
            self._stat_valid = self._stat_all = 0.0
        pred = preds["dist"][:, 0, :].clamp(LOG_MIN - 1.0, LOG_MAX + 1.0)
        tgt_log = torch.log(tgt.clamp(min=1e-3) * 15.0)
        p, t = pred[valid], tgt_log[valid]
        base = F.huber_loss(p, t, delta=self.dist_delta, reduction="none")
        w = torch.ones_like(p)
        if self.dist_tau != 0.5 or self.dist_near_weight != 1.0:
            near = tgt[valid] * 15.0 < self.dist_near_max
            if self.dist_tau != 0.5:
                # weights normalised to mean 1 so the loss scale stays comparable
                wa = torch.where(p - t > 0, (1.0 - self.dist_tau) * 2.0, self.dist_tau * 2.0)
                w = torch.where(near, wa, w)
            if self.dist_near_weight != 1.0:
                w = torch.where(near, w * self.dist_near_weight, w)
        return (base * w).sum() / n.clamp(min=1)


    def _rank_loss(self, preds, batch, fg_mask, target_gt_idx):
        """Hinge on within-image pairs: teacher-farther box must read farther.

        Uses only the ORDER of teacher distances inside one image, so per-image
        scale drift (coco) and different intrinsics (fpv) cancel out.
        Pair matrices are (bs, n_gt, n_gt) with n_gt = max GT boxes per image,
        NOT the anchor count (that would allocate bs*A*A and OOM).
        """
        bs = preds["dist"].shape[0]
        if self.dist_rank <= 0 or "batch_idx" not in batch:
            return preds["dist"].sum() * 0.0
        counts = torch.bincount(batch["batch_idx"].long(), minlength=bs)
        ngt = int(counts.max()) if counts.numel() else 0
        if ngt < 2:
            return preds["dist"].sum() * 0.0
        dev = preds["dist"].device
        padded = self._pad_dist(batch, bs, ngt)
        pred = preds["dist"][:, 0, :]
        flat = (torch.arange(bs, device=dev)[:, None] * ngt + target_gt_idx.long().clamp(min=0)).reshape(-1)
        w = fg_mask.bool().float().reshape(-1)
        sums = torch.zeros(bs * ngt, device=dev).scatter_add_(0, flat, (pred.reshape(-1) * w))
        cnts = torch.zeros(bs * ngt, device=dev).scatter_add_(0, flat, w)
        pred_gt = (sums / cnts.clamp(min=1)).view(bs, ngt)
        valid = (cnts.view(bs, ngt) > 0) & (padded >= 0)
        tl = torch.log(padded.clamp(min=1e-3) * 15.0)
        dt = tl[:, :, None] - tl[:, None, :]
        dp = pred_gt[:, :, None] - pred_gt[:, None, :]
        pair = valid[:, :, None] & valid[:, None, :] & (dt > math.log(1.5))
        npair = pair.sum()
        if int(npair) == 0:
            return preds["dist"].sum() * 0.0
        return torch.relu(0.1 - dp)[pair].sum() / npair

    def loss_with_dist(self, preds, batch):
        """5-vector (box, cls, dfl, lambda*dist, rank*order), no batch factor."""
        (fg_mask, target_gt_idx, _tb, _ap, _st), loss, items = self.get_assigned_targets_and_loss(preds, batch)
        dist_term = self.dist_lambda * self._dist_loss(preds, batch, fg_mask, target_gt_idx)
        rank_term = self.dist_rank * self._rank_loss(preds, batch, fg_mask, target_gt_idx)
        loss = torch.cat([loss, dist_term.view(1), rank_term.view(1)])
        items = {**items, "dist_loss": dist_term.detach(), "rank_loss": rank_term.detach()}
        return loss, items


class DistV2E2ELoss(DistE2ELoss):
    """E2ELoss wrapper: distance + order terms carried by one-to-many, weight-stable."""

    def __init__(self, model: torch.nn.Module):
        E2ELoss.__init__(self, model, loss_fn=DistV2DetectionLoss)

    def __call__(self, preds, batch):
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_m5, items_m = self.one2many.loss_with_dist(one2many, batch)
        loss_o3, items_o = self.one2one.loss_raw(one2one, batch)
        total = torch.zeros_like(loss_m5)
        total[:3] = loss_m5[:3] * self.o2m + loss_o3 * self.o2o
        total[3] = loss_m5[3]
        total[4] = loss_m5[4]
        bs = one2many["boxes"].shape[0]
        items = {**items_o, "dist_loss": items_m["dist_loss"], "rank_loss": items_m["rank_loss"]}
        return total * bs, items