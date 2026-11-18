"""detdist data layer: 6-column labels ``cls cx cy w h d_norm`` from labels_dist/.

Key points
----------
* The dataset reads labels from an explicit ``labels_root`` (…/labels_dist) instead
  of ``images/`` + ``labels/`` substitution, and it NEVER writes a ``.cache`` file
  (the dataset tree is read-only for us).  The validator uses this dataset too, so
  no 5-column copy of the labels has to be kept around.
* ``d_norm`` travels with the boxes as ``labels["dist"]`` (float32, (n, 1)):
  ``>= 0`` means a valid normalised distance, ``-1`` means ignore.  It is sliced
  whenever a transform filters boxes (RandomPerspective / box_candidates) and is
  carried into the batch by :class:`DistFormat` + :meth:`DistDataset.collate_fn`.
* Augmentations that break the perspective <-> distance relation are disabled:
  mosaic, mixup, cutmix, copy_paste, rotation (degrees), shear, perspective.
  Translation / scale / HSV / flips are kept, they are handled by the box filter.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ultralytics.data.augment import Compose, Format, LetterBox, RandomFlip, RandomHSV
from ultralytics.data.augment import RandomPerspective
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER
from ultralytics.utils.instance import Instances

DIST_MAX = 15.0
_SEG_RESAMPLES = 1000


def _read_dist_label(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Parse one 6-column label file -> (cls, xywh, dist, counters)."""
    stats = {"boxes": 0, "ignore": 0, "degenerate": 0, "clipped": 0}
    if not path.is_file():
        stats["missing_file"] = 1
        return (
            np.zeros((0, 1), np.float32),
            np.zeros((0, 4), np.float32),
            np.zeros((0, 1), np.float32),
            stats,
        )
    cls, boxes, dist = [], [], []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            t = s.split()
            if len(t) != 6:
                raise ValueError(
                    f"{path}:{ln}: expected 6 columns 'cls cx cy w h d_norm', got {len(t)} columns. "
                    f"Use the labels_dist tree produced by aggregate_boxes_from_npy.py."
                )
            c = int(float(t[0]))
            x, y, w, h = (float(v) for v in t[1:5])
            d = float(t[5])
            if not np.isfinite(d) or (d != -1.0 and not (0.0 <= d <= 1.0)):
                raise ValueError(f"{path}:{ln}: d_norm={d} outside {{-1}} U [0,1]")
            if w <= 0 or h <= 0:
                stats["degenerate"] += 1
                continue
            # clip to image, keep the box only if it still has area
            x1, y1 = max(0.0, x - w / 2), max(0.0, y - h / 2)
            x2, y2 = min(1.0, x + w / 2), min(1.0, y + h / 2)
            if x2 <= x1 or y2 <= y1:
                stats["degenerate"] += 1
                continue
            if (x1, y1, x2, y2) != (x - w / 2, y - h / 2, x + w / 2, y + h / 2):
                stats["clipped"] += 1
            cls.append(c)
            boxes.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
            dist.append(d)
            stats["boxes"] += 1
            if d < 0:
                stats["ignore"] += 1
    return (
        np.asarray(cls, np.float32).reshape(-1, 1),
        np.asarray(boxes, np.float32).reshape(-1, 4),
        np.asarray(dist, np.float32).reshape(-1, 1),
        stats,
    )


class DistFormat(Format):
    """``Format`` that also emits ``dist`` (n, 1) tensor."""

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        dist = labels.pop("dist", None)
        params = super().get_params(labels)
        params["dist"] = dist
        return params

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        labels = super().apply_instances(labels, params)
        n = int(labels["bboxes"].shape[0])
        dist = (params or {}).get("dist")
        if dist is None:
            dist = np.full((n, 1), -1.0, np.float32)
        dist = np.asarray(dist, dtype=np.float32).reshape(-1, 1)
        if dist.shape[0] != n:
            raise RuntimeError(f"dist/label desync after augmentation: {dist.shape[0]} distances vs {n} boxes")
        labels["dist"] = torch.from_numpy(dist)
        return labels


class DistRandomPerspective(RandomPerspective):
    """RandomPerspective that keeps ``labels["dist"]`` aligned with the surviving boxes."""

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        cls = labels["cls"]
        dist = labels.pop("dist", None)
        instances = labels.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*params["orig_shape"][::-1])

        M = params["M"]
        scale = params["scale"]
        bboxes = self.apply_bboxes(instances.bboxes, M)

        segments = instances.segments
        keypoints = instances.keypoints
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M, params["size"])
        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M, params["size"])
        new_instances = Instances(bboxes, segments, keypoints, bbox_format="xyxy", normalized=False)
        new_instances.clip(*params["size"], preserve_obb=self.preserve_obb)

        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        i = self.box_candidates(
            box1=instances.bboxes.T, box2=new_instances.bboxes.T, area_thr=0.01 if len(segments) else 0.10
        )
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        if dist is not None:
            dist = np.asarray(dist).reshape(-1, 1)
            if dist.shape[0] != len(i):
                raise RuntimeError(f"dist/label desync before filter: {dist.shape[0]} vs {len(i)}")
            labels["dist"] = dist[i]
        return labels


class DistDataset(YOLODataset):
    """YOLODataset reading 6-column distance labels from ``labels_root``."""

    format_class = DistFormat

    def __init__(self, *args: Any, labels_root: str | Path | None = None, **kwargs: Any):
        if labels_root is None:
            raise ValueError("DistDataset requires labels_root (…/PF_11.9/labels_dist)")
        self.labels_root = Path(labels_root)
        self.dist_stats = {"images": 0, "boxes": 0, "ignore": 0, "empty": 0, "degenerate": 0, "clipped": 0, "missing": 0}
        if not self.labels_root.is_dir():
            raise FileNotFoundError(f"labels_root not found: {self.labels_root}")
        super().__init__(*args, **kwargs)

    # --------------------------------------------------------------- label loading
    def _label_path_for(self, im_file: str) -> Path:
        parts = list(Path(im_file).parts)
        if "images" not in parts:
            raise ValueError(f"image path has no 'images' component: {im_file}")
        idx = len(parts) - 1 - parts[::-1].index("images")
        rel = Path(*parts[idx + 1 :]).with_suffix(".txt")
        return self.labels_root / rel

    @staticmethod
    def _image_shape(im_file: str) -> tuple[int, int]:
        with Image.open(im_file) as im:  # header only, no full decode
            w, h = im.size
        return int(h), int(w)

    def get_labels(self) -> list[dict]:  # noqa: D102 - no cache file is written on purpose
        labels: list[dict] = []
        for im_file in self.im_files:
            lp = self._label_path_for(im_file)
            try:
                cls, bboxes, dist, st = _read_dist_label(lp)
            except ValueError as e:
                raise ValueError(f"bad distance label: {e}") from e
            h, w = self._image_shape(im_file)
            self.dist_stats["images"] += 1
            self.dist_stats["boxes"] += st["boxes"]
            self.dist_stats["ignore"] += st["ignore"]
            self.dist_stats["degenerate"] += st["degenerate"]
            self.dist_stats["clipped"] += st["clipped"]
            self.dist_stats["missing"] += st.get("missing_file", 0)
            n = len(cls)
            if n == 0:
                self.dist_stats["empty"] += 1
            labels.append(
                {
                    "im_file": im_file,
                    "shape": (h, w),
                    "cls": cls,
                    "bboxes": bboxes,
                    "segments": np.zeros((0, _SEG_RESAMPLES, 2), np.float32),
                    "keypoints": None,
                    "normalized": True,
                    "bbox_format": "xywh",
                    "dist": dist,
                }
            )
        LOGGER.info(
            f"{self.prefix}distant labels: root={self.labels_root} images={self.dist_stats['images']} "
            f"boxes={self.dist_stats['boxes']} ignore={self.dist_stats['ignore']} "
            f"empty={self.dist_stats['empty']} degenerate={self.dist_stats['degenerate']} "
            f"clipped={self.dist_stats['clipped']} missing={self.dist_stats['missing']} "
            f"(label path is NOT substituted from images/)"
        )
        return labels

    def update_labels_info(self, label: dict) -> dict:  # noqa: D102
        dist = label.pop("dist", None)
        out = super().update_labels_info(label)
        n = len(out["instances"])
        if dist is None:
            dist = np.full((n, 1), -1.0, np.float32)
        dist = np.asarray(dist, dtype=np.float32).reshape(-1, 1)
        if dist.shape[0] != n:
            raise RuntimeError(f"dist/instances desync: {dist.shape[0]} vs {n}")
        out["dist"] = dist
        return out

    # ------------------------------------------------------------------ transforms
    def build_transforms(self, hyp):  # noqa: D102
        imgsz = self.imgsz
        if isinstance(imgsz, (tuple, list)):
            h, w = int(imgsz[0]), int(imgsz[-1])
        else:
            h = w = int(imgsz)
        if self.augment:
            hyp.mosaic = hyp.mixup = hyp.cutmix = hyp.copy_paste = 0.0
            hyp.degrees = hyp.shear = hyp.perspective = 0.0
            transforms = Compose(
                [
                    DistRandomPerspective(
                        degrees=0.0,
                        translate=hyp.translate,
                        scale=hyp.scale,
                        shear=0.0,
                        perspective=0.0,
                        size=(w, h),
                        preserve_obb=getattr(self, "use_obb", False),
                    ),
                    RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
                    RandomFlip(direction="vertical", p=hyp.flipud),
                    RandomFlip(direction="horizontal", p=hyp.fliplr),
                ]
            )
        else:
            transforms = Compose([LetterBox(new_shape=(h, w), scaleup=False)])
        transforms.append(
            self.format_class(
                bbox_format="xywh",
                normalize=True,
                return_mask=False,
                return_keypoint=False,
                return_obb=False,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,
            )
        )
        return transforms

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:  # noqa: D102
        new_batch = YOLODataset.collate_fn(batch)
        if "dist" in new_batch:
            new_batch["dist"] = torch.cat(list(new_batch["dist"]), 0)
        return new_batch

class GenDataset(DistDataset):
    """DistDataset + optional BackgroundSwap (scene-identity breaking, train only)."""

    def build_transforms(self, hyp):
        transforms = super().build_transforms(hyp)
        p = float(os.environ.get("DETDIST_V2_BGSWAP", "0"))
        if self.augment and p > 0:
            from metdist.augment import BackgroundSwap
            transforms.transforms.insert(0, BackgroundSwap(self, p=p))
        return transforms
