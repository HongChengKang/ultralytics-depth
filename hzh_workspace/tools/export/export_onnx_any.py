# -*- coding: utf-8 -*-
"""按流程导出任意 .pt：python export_onnx_any.py <weights.pt> <imgsz> <out.onnx>"""
import os, sys
from pathlib import Path

ROOT = Path("/home/kang/workspace/projects/yolo-depth/ultralytics-depth")
sys.path.insert(0, str(ROOT / "hck_workspace" / "detdepth"))
sys.path.insert(0, str(ROOT))

import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Detect, Depth, RTDETRDecoder
from ultralytics.utils.export.engine import _NormalizeCoords, torch2onnx

WEIGHTS = Path(sys.argv[1])
IMGSZ = int(sys.argv[2])
ONNX_PATH = Path(sys.argv[3])


class RKNNDetDepth(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        det, preds = out if isinstance(out, (tuple, list)) else (out, {})
        return det, preds["depth"]


def main():
    model = YOLO(str(WEIGHTS))
    head = model.model
    for p in head.parameters():
        p.requires_grad = False
    head.eval(); head.float()
    head = head.fuse(imgsz=(IMGSZ, IMGSZ))
    for m in head.modules():
        if isinstance(m, (Detect, RTDETRDecoder)):
            m.dynamic = False; m.export = True; m.format = "rknn"; m.shape = None
        elif isinstance(m, Depth):
            m.export = True; m.format = "rknn"
    nc = len(head.names)
    net = _NormalizeCoords(RKNNDetDepth(head), IMGSZ, IMGSZ, "detect", nc, None)
    net.eval()
    im = torch.zeros(1, 3, IMGSZ, IMGSZ)
    with torch.no_grad():
        det, depth = net(im)
    print("forward", tuple(det.shape), tuple(depth.shape), "imgsz", IMGSZ)
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch2onnx(net, im, ONNX_PATH, opset=12, input_names=["images"],
               output_names=["output0", "depth"], dynamic=None)
    print("saved", ONNX_PATH, ONNX_PATH.stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    main()
