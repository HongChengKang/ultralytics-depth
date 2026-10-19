"""YOLO26 joint detection + depth: one shared backbone/neck, a Detect head and a Depth head on the same P3-P5 features.

Everything here subclasses existing ultralytics components; the ultralytics package itself is unchanged. Checkpoints
pickle these classes by module name ``detdepth``, so this directory must be on ``sys.path`` when loading them (the
train.py/predict.py entry points next to this file guarantee that).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from ultralytics.data.augment import DepthFormat, Format
from ultralytics.data.dataset import DepthDataset, YOLODataset
from ultralytics.data.utils import get_split_fraction, load_depth
from ultralytics.models.yolo.detect import DetectionPredictor, DetectionTrainer
from ultralytics.nn.modules.head import Depth, Detect
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import LOGGER, RANK, colorstr, ops
from ultralytics.utils.loss import DepthLoss26, E2ELoss
from ultralytics.utils.torch_utils import initialize_weights, unwrap_model


class DetectDepth(Detect):
    """Detect head that also runs a Depth decoder (``dhead``) on the same multi-scale features.

    Training returns the Detect prediction dict plus a ``"depth"`` key (B, 1, H/4, W/4). Inference returns the usual
    Detect ``(y, preds)`` tuple with ``preds["depth"]`` holding the calibrated depth map, so detection post-processing
    (NMS, validator, predictor) is unchanged.
    """

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: tuple = (), c_mid: int = 256):
        """Initialize Detect branches and the depth decoder."""
        super().__init__(nc, reg_max, end2end, ch)
        self.dhead = Depth(c_mid, ch)

    def forward(self, x: list[torch.Tensor]):
        """Run detection and depth on the shared features."""
        depth = self.dhead(x)
        out = super().forward(x)
        if self.training:
            return {**out, "depth": depth["depth"]}
        y, preds = out if isinstance(out, tuple) else (out, {})
        return y, {**preds, "depth": depth}


class DetDepthModel(DetectionModel):
    """DetectionModel built from a standard detect YAML whose Detect head is swapped for DetectDepth."""

    def __init__(self, cfg="yolo26n.yaml", ch=3, nc=None, verbose=True):
        """Build the detection model normally (strides, bias init), then replace its head in place."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=False)
        old = self.model[-1]
        new = DetectDepth(
            old.nc, old.reg_max, getattr(old, "one2one_cv2", None) is not None, [m[0].conv.in_channels for m in old.cv2]
        )
        initialize_weights(new)  # BatchNorm eps/momentum are not in state_dict
        new.load_state_dict(old.state_dict(), strict=False)  # Detect weights incl. bias_init; dhead stays fresh
        for k in ("f", "i", "np", "stride", "inplace", "_end2end"):
            if hasattr(old, k):
                setattr(new, k, getattr(old, k))
        new.type = "DetectDepth"
        self.model[-1] = new
        if verbose:
            self.info()
            LOGGER.info("")

    def load_depth_head(self, weights: str | Path) -> None:
        """Copy the Depth head (last layer) of a yolo26*-depth checkpoint into ``dhead``."""
        src = load_checkpoint(weights)[0].model[-1]
        # released depth weights carry an unused extra refine block, so only missing keys are an error
        missing = self.model[-1].dhead.load_state_dict(src.float().state_dict(), strict=False).missing_keys
        if missing:
            raise RuntimeError(f"Depth head weights in {weights} are missing {missing}")
        LOGGER.info(f"Transferred depth head from {weights}")

    def init_criterion(self):
        """Return the joint detection + depth criterion."""
        return DetDepthLoss(self)


class DetDepthLoss(E2ELoss):
    """YOLO26 end-to-end detection loss plus DepthLoss26 on ``preds["depth"]`` (weighted by dlog/dgrad args)."""

    def __init__(self, model: torch.nn.Module):
        """Initialize detection and depth criteria."""
        super().__init__(model)
        self.depth = DepthLoss26(model)

    def __call__(self, preds, batch):
        """Return concatenated (box, cls, dfl, dlog, dgrad) losses and their merged loss-item dict."""
        det_loss, det_items = super().__call__(preds, batch)
        depth_loss, depth_items = self.depth(self.one2many.parse_output(preds)["depth"], batch)
        return torch.cat((det_loss, depth_loss)), {**det_items, **depth_items}


class DetDepthFormat(Format):
    """Detection Format that also tensorizes the paired depth map, keeping instance labels."""

    apply_depth = DepthFormat.apply_depth


class DetDepthDataset(YOLODataset):
    """YOLO detection dataset that also loads images/<split>/x.jpg -> depth/<split>/x.png pseudo depth."""

    format_class = DetDepthFormat
    _depth_path_for = DepthDataset._depth_path_for

    def get_image_and_label(self, index: int) -> dict:
        """Load image, box labels, and the depth map resized to the loaded image size."""
        label = super().get_image_and_label(index)
        depth = load_depth(self._depth_path_for(label["im_file"]), self.data.get("depth_scale", 1000))
        h, w = label["resized_shape"]
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        label["depth"] = depth
        return label

    def build_transforms(self, hyp):
        """Disable mix transforms that do not carry depth (Mosaic, MixUp, CutMix, CopyPaste)."""
        hyp.mosaic = hyp.mixup = hyp.cutmix = hyp.copy_paste = 0.0
        return super().build_transforms(hyp)


class DetDepthTrainer(DetectionTrainer):
    """DetectionTrainer for DetDepthModel on DetDepthDataset.

    Set ``DetDepthTrainer.depth_weights`` to a yolo26*-depth checkpoint to initialize the depth head (fresh runs only).
    """

    depth_weights = None

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build a DetDepthDataset with the same settings DetectionTrainer uses for YOLODataset."""
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        return DetDepthDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=get_split_fraction(self.args.fraction, "train" if mode == "train" else self.args.split),
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a DetDepthModel with detection weights and (optionally) a pretrained depth head."""
        model = self.set_model_names_for_load(
            DetDepthModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)
        if self.depth_weights and not self.args.resume:
            model.load_depth_head(self.depth_weights)
        return model

    def preprocess_batch(self, batch: dict) -> dict:
        """Normalize images and cast depth to float32."""
        batch = super().preprocess_batch(batch)
        batch["depth"] = batch["depth"].float()
        return batch


class DetDepthPredictor(DetectionPredictor):
    """DetectionPredictor that attaches the dense depth map and per-box depth (``Results.box_depth``)."""

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Run detection post-processing, then map depth to original images and take each box's median depth."""
        depth = preds[1]["depth"]
        results = super().postprocess(preds, img, orig_imgs, **kwargs)
        depth = F.interpolate(depth.float(), size=img.shape[2:], mode="bilinear", align_corners=True)
        for r, d in zip(results, depth):
            d = ops.scale_masks(d[None], r.orig_shape)[0, 0]  # remove letterbox padding, resize to original
            r.update(depth=d)
            r.box_depth = box_depth(d, r.boxes.xyxy)
        return results


def box_depth(depth: torch.Tensor, xyxy: torch.Tensor) -> torch.Tensor:
    """Return the median depth inside each (x1, y1, x2, y2) box of an (H, W) depth map."""
    h, w = depth.shape
    out = torch.zeros(len(xyxy), device=depth.device)
    for j, (x1, y1, x2, y2) in enumerate(xyxy.round().long().tolist()):
        x1, x2, y1, y2 = max(x1, 0), min(x2, w), max(y1, 0), min(y2, h)
        if x2 > x1 and y2 > y1:
            out[j] = depth[y1:y2, x1:x2].median()
        else:  # degenerate box: sample its (clamped) center
            out[j] = depth[min(max((y1 + y2) // 2, 0), h - 1), min(max((x1 + x2) // 2, 0), w - 1)]
    return out
