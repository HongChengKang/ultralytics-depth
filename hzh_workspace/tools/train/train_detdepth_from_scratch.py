"""From-scratch YOLO26 detection+depth entry for COCO_PF (thin wrapper, does NOT modify train.py).

Difference from hck_workspace/detdepth/train.py: ``DetDepthTrainer.depth_weights = None`` so the depth head is
randomly initialized instead of loading model_common/yolo26n-depth.pt. Everything else is the same
YOLO(<yaml>).train(trainer=DetDepthTrainer, **args) call.

Usage: ./hzh_workspace/train_coco_pf_scratch.sh [key=value ...]
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]   # HERE=.../hzh_workspace/tools/train; parents[2]=ultralytics-depth
sys.path[:0] = [str(REPO / "hck_workspace" / "detdepth"), str(REPO)]

from detdepth import DetDepthTrainer  # noqa: E402

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import smart_value  # noqa: E402

CFG = REPO / "hzh_workspace" / "configs" / "detdepth_from_scratch.yaml"
DATA = REPO / "hzh_workspace" / "configs" / "coco_pf_detdepth.yaml"

if __name__ == "__main__":
    args = dict(
        data=str(DATA),
        epochs=500,
        imgsz=768,
        batch=32,
        device="0",
        warmup_epochs=3.0,
        patience=50,
        lr0=0.001,
        project=str(REPO / "hzh_workspace" / "runs"),
        name="detdepth_coco_pf_scratch",
    )
    args.update((k, smart_value(v)) for k, v in (a.split("=", 1) for a in sys.argv[1:]))
    DetDepthTrainer.depth_weights = None  # from scratch: no pretrained depth head
    YOLO(str(CFG)).train(trainer=DetDepthTrainer, **args)
