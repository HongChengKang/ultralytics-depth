"""Export a detdepth / dist-plan-a .pt to ONNX (rknn_model_zoo YOLOv8 split layout).

Two head flavours are supported, chosen automatically from the checkpoint:

A) DetDepth (旧的检测+深度头, runs/detdepth_*)
   per level P3/P4/P5: box_p{3,4,5} (B,4,H,W), cls_p{3,4,5} (B,nc,H,W), score_sum_p{3,4,5}
   then depth (B,1,imgsz_h,imgsz_w) — relative depth, same as the merged export.

B) DetectDist (detdist 距离回归头, runs/dist_pf_all)
   per level P3/P4/P5: box_p{3,4,5}, cls_p{3,4,5}, score_sum_p{3,4,5}
   then dist_p{3,4,5} (B,1,H_s,W_s) — **sigmoid(raw) * 15 = 米**，已按级归一化到 [0,15]

Decode (both flavours): cx = (j + 0.5) * s, cy = (i + 0.5) * s;
x1, y1, x2, y2 = cx - l*s, cy - t*s, cx + r*s, cy + b*s.

Usage:
    python tools/split_pipeline_detdist/export_onnx_split.py <best.pt> <out.onnx> [--imgsz 480x320] [--batch 1]

--imgsz must match training (ultralytics convention: 480x320 = height 480 x width 320);
both sides must be multiples of 32 (the P5 stride).
then convert with convert_rknn.py <out.onnx> <out.rknn> --platform rk3588 [--fp].
"""

import argparse
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path('/home/kang/workspace/projects/yolo-depth/ultralytics-depth')
WS = REPO / 'hzh_workspace'
sys.path[:0] = [
    str(HERE),
    str(REPO),
    str(REPO / 'hck_workspace' / 'detdepth'),
    str(WS / 'libs'),  # detdist.head.DetectDist（detdist 的 checkpoint pickle 需要它）
]

import torch  # noqa: E402

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import Depth  # noqa: E402
from ultralytics.utils.export.engine import torch2onnx  # noqa: E402

LEVELS = ("p3", "p4", "p5")
DETDIST_MAX_M = float(os.environ.get("DETDIST_MAX_M", "15.0"))  # 与训练侧 d_m = d_norm * max_dist 一致


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


def split_forward_dist(self, x):
    """DetectDist forward: same 3 tensors per level, then one metric-distance map per level (metres)."""
    outs = []
    for i in range(self.nl):
        cls = self.cv3[i](x[i]).sigmoid()
        outs += [self.cv2[i](x[i]), cls, cls.sum(1, keepdim=True).clamp(0, 1)]
    outs += [(self.dist[i](x[i]).sigmoid() * DETDIST_MAX_M) for i in range(self.nl)]
    return tuple(outs)


def main(opt):
    model = YOLO(opt.weights).model.float().eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.fuse()
    head = model.model[-1]
    has_dist = hasattr(head, "dist") and len(getattr(head, "dist")) == len(LEVELS)
    head.forward = types.MethodType(split_forward_dist if has_dist else split_forward, head)
    if has_dist:
        for m in model.modules():
            if isinstance(m, Depth):  # 旧头才有的模块；detdist 没有，保险起见
                m.export = True
        names = [f"{k}_{lv}" for lv in LEVELS for k in ("box", "cls", "score_sum")] + [f"dist_{lv}" for lv in LEVELS]
        print(f"head: {type(head).__name__} (distance branch x{len(LEVELS)}, d_m = sigmoid * {DETDIST_MAX_M:g})")
    else:
        for m in model.modules():
            if isinstance(m, Depth):
                m.export = True  # upsample depth to input size inside the graph
        names = [f"{k}_{lv}" for lv in LEVELS for k in ("box", "cls", "score_sum")] + ["depth"]
        print(f"head: {type(head).__name__} (depth map head)")
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
    p.add_argument("--imgsz", default="480x320", help="训练时的输入尺寸：640 或 高x宽（如 480x320），需为 32 的倍数")
    p.add_argument("--batch", type=int, default=1)
    main(p.parse_args())
