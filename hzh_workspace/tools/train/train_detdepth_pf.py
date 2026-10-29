#!/home/kang/miniforge3/envs/yolo/bin/python
"""COCO_PF_dataset YOLO26 detection+depth entry — mirrors hck_workspace/detdepth/train.py.

- Detection: starts from the yolo26n architecture YAML (random init, no yolo26n.pt).
- Depth head: initialized from model_common/yolo26n-depth.pt (same as train.py's DetDepthTrainer.depth_weights).
- Depth supervision: the teacher-generated depth/train, depth/val maps already inside /mnt/data2/COCO_PF.
- Default device=0 (single A6000, batch=80). For DDP use device="0,1"; batch is split across cards.

Usage: ./hzh_workspace/train_coco_pf.sh [key=value ...]   # foreground
       ./hzh_workspace/train_coco_pf.sh start           # detached, survives SSH logout
       ./hzh_workspace/train_coco_pf.sh status|tail|stop
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
MC = REPO / "model_common"

if __name__ == "__main__":
    args = dict(
        data=str(DATA),
        epochs=400,
        imgsz=640,
        batch=80,  # total batch, 128 per GPU on 2x RTX A6000 (48 GB) -> ~30 GB VRAM per card (ceiling ~384)
        device="0",
        warmup_epochs=3.0,
        patience=20,
        lr0=0.001,
        project=str(REPO / "hzh_workspace" / "runs"),
        name="detdepth_coco_pf",  # v2 正式；v1 旧结果在 runs/detdepth_coco_pf_old
    )
    args.update((k, smart_value(v)) for k, v in (a.split("=", 1) for a in sys.argv[1:]))
    DetDepthTrainer.depth_weights = str(MC / "yolo26n-depth.pt")  # same as train.py
    YOLO(str(CFG)).train(trainer=DetDepthTrainer, **args)
