"""detdist_v2 trainer: v1 data path + two-source warm start.

  * dataset: detdist.data.DistDataset (unchanged), labels_root defaults to
    <dataset_root>/labels_dist_own (coco/fpv distance masked to -1, detection
    supervision intact);
  * warm start: backbone (model.0..model.10) from a metric-depth checkpoint
    (model_common/yolo26n-depth.pt), remaining matching weights from a
    detection checkpoint (runs/detdist_pf_cal/weights/best.pt); the new
    distance towers always start fresh. Both sources are optional.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from ultralytics.engine.trainer import RANK
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import unwrap_model

from metdist.data import GenDataset
from metdist.model import DistV2Model

DEFAULT_LABELS_ROOT = "/mnt/data2/hzh/PF_11.9/labels_dist_own"
BACKBONE_IDS = range(0, 11)  # yaml layers 0..10


def _is_backbone(key: str) -> bool:
    if not key.startswith("model."):
        return False
    part = key.split(".", 2)[1]
    return part.isdigit() and int(part) in BACKBONE_IDS


def _ckpt_state(weights) -> dict:
    if isinstance(weights, dict):
        m = weights.get("ema") or weights.get("model")
    else:
        m = weights
    return m.float().state_dict()


class DistV2Trainer(DetectionTrainer):
    labels_root: str = os.environ.get("DETDIST_LABELS_ROOT", DEFAULT_LABELS_ROOT)
    depth_weights: str = ""  # set by tools/train/train_detdist_v2.py
    det_weights: str = ""

    def build_dataset(self, img_path, mode="train", batch=None):  # noqa: D102
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        if not Path(self.labels_root).is_dir():
            raise FileNotFoundError(
                f"labels_root not found: {self.labels_root} "
                f"(run tools/data/make_own_dist_labels.py, or pass --labels_root)"
            )
        return GenDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction,
            labels_root=self.labels_root,
        )

    def _warm_start(self, model: DistV2Model) -> None:
        msd = model.state_dict()
        depth_loaded = 0
        if self.depth_weights:
            csd = _ckpt_state(torch.load(self.depth_weights, map_location="cpu", weights_only=False))
            take = {k: v for k, v in csd.items() if _is_backbone(k) and k in msd and msd[k].shape == v.shape}
            msd.update(take)
            depth_loaded = len(take)
            LOGGER.info(f"[detdist_v2] backbone <- {self.depth_weights}: {depth_loaded} tensors")
        if self.det_weights:
            csd = _ckpt_state(torch.load(self.det_weights, map_location="cpu", weights_only=False))
            take = {
                k: v
                for k, v in csd.items()
                if k in msd and msd[k].shape == v.shape and ".dist." not in k
                and not (depth_loaded and _is_backbone(k))
            }
            msd.update(take)
            LOGGER.info(f"[detdist_v2] neck/detect <- {self.det_weights}: {len(take)} tensors")
        model.load_state_dict(msd)

    def validate(self):
        """Distance-aware fitness: mAP50-95 - 2 * val/dist_loss.

        The stock fitness is detection mAP only; with a detection warm start it
        peaks at epoch ~2 and early-stops while the distance head is still
        improving (observed: best=ep2, val/dist kept dropping until ep32).
        """
        if self.ema and self.world_size > 1:
            from ultralytics.utils import dist

            dist.broadcast(self.ema.ema.buffers(), src=0)
        metrics = self.validator(self)
        if metrics is None:
            return None, None
        metrics.pop("fitness", None)
        fitness = float(metrics.get("metrics/mAP50-95(B)", 0.0)) - 2.0 * float(
            metrics.get("val/dist_loss", 0.0)
        )
        if self.best_fitness is None or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):  # noqa: D102
        model = DistV2Model(
            cfg,
            ch=self.data.get("channels", 3),
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )
        model = self.set_model_names_for_load(model)
        if self.depth_weights or self.det_weights:
            self._warm_start(model)
        elif weights:
            model.load(weights)
        return model