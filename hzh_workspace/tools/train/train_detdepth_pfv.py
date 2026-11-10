"""PFV_dataset YOLO26 detection+depth entry — mirrors hck_workspace/detdepth/train.py.

- Detection: starts from the yolo26n architecture YAML (random init, no yolo26n.pt).
- Depth head: initialized from model_common/yolo26n-depth.pt (same as train.py's DetDepthTrainer.depth_weights).
- Depth supervision: the teacher-generated depth/train, depth/val maps already inside /mnt/data2/PFV_dataset.

Usage: ./hzh_workspace/train_pfv_dataset.sh [key=value ...]
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
DATA = REPO / "hzh_workspace" / "configs" / "pfv_detdepth.yaml"
MC = REPO / "model_common"

if __name__ == "__main__":
    args = dict(
        data=str(DATA),
        epochs=500,
        imgsz=640,
        batch=64,
        device="0",
        warmup_epochs=3.0,
        patience=50,
        lr0=0.001,
        project=str(REPO / "hzh_workspace" / "runs"),
        name="detdepth_pfv_dataset",
    )
    args.update((k, smart_value(v)) for k, v in (a.split("=", 1) for a in sys.argv[1:]))
    DetDepthTrainer.depth_weights = str(MC / "yolo26n-depth.pt")  # same as train.py
    YOLO(str(CFG)).train(trainer=DetDepthTrainer, **args)
