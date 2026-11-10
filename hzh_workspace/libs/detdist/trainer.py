"""detdist trainer: the reference pipeline (train_detdepth_pf.py) plus the distance branch.

What differs from the reference trainer:
  * ``get_model`` builds :class:`detdist.model.DistModel` (stock YAML + one 1x1
    distance conv per level),
  * ``build_dataset`` feeds **both** splits from ``<dataset_root>/labels_dist`` via
    :class:`detdist.data.DistDataset` (6-column labels, ``get_labels`` is custom so
    no ``.cache`` file is ever written into the dataset tree),
  * the criterion comes from ``DistModel.init_criterion`` (detection + Huber dist).

There is no label mirror: the data yaml points straight at
``/mnt/data2/hzh/PF_11.9/images/{train,val}`` and the label tree is resolved
separately through ``labels_root``.
"""

from __future__ import annotations

import os
from pathlib import Path

from ultralytics.engine.trainer import RANK
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import unwrap_model

from detdist.data import DistDataset
from detdist.model import DistModel

DEFAULT_LABELS_ROOT = "/mnt/data2/hzh/PF_11.9/labels_dist"


class DistTrainer(DetectionTrainer):
    """DetectionTrainer that adds box-level distance supervision."""

    #: overwritten by tools/train/train_detdist.py (--labels_root / dataset yaml)
    labels_root: str = os.environ.get("DETDIST_LABELS_ROOT", os.environ.get("DIST_LABELS_ROOT", DEFAULT_LABELS_ROOT))

    def build_dataset(self, img_path, mode="train", batch=None):  # noqa: D102
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        if not Path(self.labels_root).is_dir():
            raise FileNotFoundError(
                f"labels_root not found: {self.labels_root} "
                f"(run tools/data/make_dist_labels.py, or pass --labels_root)"
            )
        return DistDataset(
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

    def get_model(self, cfg=None, weights=None, verbose=True):  # noqa: D102
        model = DistModel(
            cfg,
            ch=self.data.get("channels", 3),
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )
        model = self.set_model_names_for_load(model)
        if weights:
            model.load(weights)
        return model
