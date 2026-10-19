"""Run the detection + depth model: boxes with per-box (relative) depth.

Usage:
    python hck_workspace/detdepth/predict.py --weights runs/detdepth/visdrone/weights/best.pt --source path/to/images

Saves <out>/<stem>.jpg (boxes labeled "cls conf | d=<median depth>") and <out>/<stem>.txt
("cls conf x1 y1 x2 y2 depth" per box, pixel coordinates of the original image).
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO)]  # this checkout's ultralytics; checkpoints pickle classes from `detdepth`

import cv2  # noqa: E402
from detdepth import DetDepthPredictor  # noqa: E402

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.plotting import Annotator, colors  # noqa: E402


def main(opt):
    out = Path(opt.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(opt.weights)
    results = model.predict(
        opt.source, predictor=DetDepthPredictor, imgsz=opt.imgsz, conf=opt.conf, device=opt.device, stream=True
    )
    for r in results:
        stem = Path(r.path).stem
        ann = Annotator(r.orig_img.copy(), line_width=2)
        lines = []
        for (x1, y1, x2, y2), conf, c, d in zip(
            r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.int().tolist(), r.box_depth.tolist()
        ):
            ann.box_label((x1, y1, x2, y2), f"{r.names[c]} {conf:.2f} | d={d:.1f}", color=colors(c, True))
            lines.append(f"{c} {conf:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {d:.3f}\n")
        cv2.imwrite(str(out / f"{stem}.jpg"), ann.result())
        (out / f"{stem}.txt").write_text("".join(lines))
        print(f"{stem}: {len(lines)} boxes")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights",  default="../runs/detdepth/visdrone/weights/best.pt")
    p.add_argument("--source", default="../../data/images/test")
    p.add_argument("--out", default=str(REPO / "runs" / "detdepth" / "predict"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="0")
    main(p.parse_args())
