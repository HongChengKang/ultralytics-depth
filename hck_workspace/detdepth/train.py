"""Train the YOLO26 detection + depth joint model.

Usage:
    python hck_workspace/detdepth/train.py epochs=100 imgsz=640 batch=16 device=0
    python hck_workspace/detdepth/train.py resume=True model=runs/detdepth/visdrone/weights/last.pt

Any ultralytics train argument can be passed as key=value. Depth loss weights are the standard dlog/dgrad args.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO)]  # this checkout's ultralytics; checkpoints pickle classes from `detdepth`

from detdepth import DetDepthTrainer  # noqa: E402

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import smart_value  # noqa: E402

MC = REPO / "model_common"

if __name__ == "__main__":
    args = dict(
        model=str(MC / "yolo26n.pt"),
        data=str(HERE / "visdrone-detdepth.yaml"),
        epochs=100,
        imgsz=640,
        batch=16,
        project=str(REPO / "runs" / "detdepth"),
        name="visdrone",
    )
    args.update((k, smart_value(v)) for k, v in (a.split("=", 1) for a in sys.argv[1:]))
    DetDepthTrainer.depth_weights = str(MC / "yolo26n-depth.pt")
    model = args.pop("model")
    YOLO(model).train(trainer=DetDepthTrainer, **args)
