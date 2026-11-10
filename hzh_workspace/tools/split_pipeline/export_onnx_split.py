"""Export a detdepth .pt to ONNX with per-scale detection outputs (rknn_model_zoo YOLOv8 layout) plus depth.

Outputs, for each level P3/P4/P5 (stride 8/16/32, grid H x W = imgsz/stride):
    box_p{3,4,5}       (B, 4, H, W)   raw l, t, r, b distances in grid units (YOLO26 is DFL-free, reg_max=1)
    cls_p{3,4,5}       (B, nc, H, W)  sigmoid class scores
    score_sum_p{3,4,5} (B, 1, H, W)   clamp(sum of class scores, 0, 1), for fast filtering
then
    depth              (B, 1, imgsz_h, imgsz_w) relative depth, same as the merged export

Decode: cx = (j + 0.5) * s, cy = (i + 0.5) * s; x1, y1, x2, y2 = cx - l*s, cy - t*s, cx + r*s, cy + b*s.

Usage:
    python hck_workspace/detdepth/export_onnx_split.py <best.pt> <out.onnx> [--imgsz 640 | 480x320] [--batch 4]

--imgsz must match the training size (ultralytics convention: 480x320 = height 480 x width 320); both
sides must be multiples of 32 (the P5 stride).
then convert with convert_rknn.py <out.onnx> <out.rknn> --platform rk3576 [--fp].
"""

import argparse
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path('/home/kang/workspace/projects/yolo-depth/ultralytics-depth')
sys.path[:0] = [str(HERE), str(REPO), '/home/kang/workspace/projects/yolo-depth/ultralytics-depth/hck_workspace/detdepth']  # this checkout's ultralytics; checkpoints pickle classes from `detdepth`

import torch  # noqa: E402

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import Depth  # noqa: E402
from ultralytics.utils.export.engine import torch2onnx  # noqa: E402

LEVELS = ("p3", "p4", "p5")


def parse_imgsz(s) -> tuple:
    """Parse --imgsz into (h, w): 640 -> (640, 640), "480x320" -> (480, 320); both must be multiples of 32."""
    parts = str(s).lower().replace(",", "x").replace(" ", "").split("x")
    h = int(parts[0])
    w = int(parts[1]) if len(parts) > 1 else h
    if h % 32 or w % 32:
        raise SystemExit(f"--imgsz 两边都需为 32 的倍数（P5 stride），当前: {h}x{w}")
    return h, w


def split_forward(self, x):
    """DetectDepth forward returning raw per-level box/cls/score_sum tensors and the depth map."""
    outs = []
    for i in range(self.nl):
        cls = self.cv3[i](x[i]).sigmoid()
        outs += [self.cv2[i](x[i]), cls, cls.sum(1, keepdim=True).clamp(0, 1)]
    return (*outs, self.dhead(x))


def main(opt):
    model = YOLO(opt.weights).model.float().eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.fuse()
    head = model.model[-1]
    head.forward = types.MethodType(split_forward, head)
    for m in model.modules():
        if isinstance(m, Depth):
            m.export = True  # upsample depth to input size inside the graph
    names = [f"{k}_{lv}" for lv in LEVELS for k in ("box", "cls", "score_sum")] + ["depth"]
    h, w = parse_imgsz(opt.imgsz)
    im = torch.zeros(opt.batch, 3, h, w)
    print(f"input {tuple(im.shape)} (batch {opt.batch}, imgsz {h}x{w})")
    with torch.no_grad():
        outs = model(im)
    for n, o in zip(names, outs):
        print(f"{n:14s} {tuple(o.shape)}")
    out = Path(opt.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch2onnx(model, im, out, opset=12, input_names=["images"], output_names=names, dynamic=None)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB), classes {model.names}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("weights")
    p.add_argument("out")
    p.add_argument("--imgsz", default="640", help="训练时的输入尺寸：640 或 高x宽（如 480x320），需为 32 的倍数")
    p.add_argument("--batch", type=int, default=4)
    main(p.parse_args())
