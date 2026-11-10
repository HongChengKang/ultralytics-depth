# -*- coding: utf-8 -*-
"""
pt -> onnx（rknn 格式，2 输出）

参照 ultralytics/engine/exporter.py 里 format='rknn' + quantize=8 的导出路径改写：
  1. eval + fuse + 给 Detect / Depth 打上 export=True, format='rknn'
  2. 自定义 RKNNDetDepth 把 DetectDepth 的 (y, {'depth': ...}) 摊平成 (det, depth)
  3. 套 _NormalizeCoords，把 xywh 归一化到 0~1（int8 per-tensor 量化必需）
  4. torch2onnx 导出：images [1,3,640,640] -> output0 [1,5,8400], depth [1,1,640,640]

用法（必须在 yolo 环境，rknn 环境没有 torch/ultralytics/PIL）：
    conda activate yolo
    python hzh_workspace/tools/pt_to_onnx.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # ultralytics-depth/
DETDEPTH_DIR = ROOT / "hck_workspace" / "detdepth"  # 自定义 DetectDepth 模块所在目录

# 注意：本脚本以 hzh_workspace/tools/ 为 sys.path[0] 启动，默认会 import 到
# site-packages 里的旧版 ultralytics(8.4.142)；这里显式把仓库根目录插到最前，
# 保证用的是训练该权重时的仓库版本(8.4.178)。
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DETDEPTH_DIR))

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.nn.modules import Detect, Depth, RTDETRDecoder
from ultralytics.utils.export.engine import _NormalizeCoords, torch2onnx

WEIGHTS = ROOT / "runs/detdepth/visdrone/weights/best.pt"
ONNX_PATH = ROOT / "hzh_workspace/weights/best.onnx"

IMGSZ = 640
OPSET = 12


class RKNNDetDepth(torch.nn.Module):
    """把 DetectDepth 导出模式的 (det, {'depth': depth}) 摊平成 (det, depth)。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        det, preds = out if isinstance(out, (tuple, list)) else (out, {})
        return det, preds["depth"]


def main():
    print(f"ultralytics: {ultralytics.__version__} @ {ultralytics.__file__}")
    print(f"weights    : {WEIGHTS}")
    if not WEIGHTS.is_file():
        sys.exit(f"找不到权重: {WEIGHTS}")
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(WEIGHTS))
    head = model.model

    # ---- 与 exporter.py:901-940 的预处理保持一致 ----
    for p in head.parameters():
        p.requires_grad = False
    head.eval()
    head.float()
    head = head.fuse(imgsz=(IMGSZ, IMGSZ))
    for m in head.modules():
        if isinstance(m, (Detect, RTDETRDecoder)):
            m.dynamic = False
            m.export = True
            m.format = "rknn"
            m.shape = None
        elif isinstance(m, Depth):
            m.export = True
            m.format = "rknn"

    nc = len(head.names)
    net = _NormalizeCoords(RKNNDetDepth(head), IMGSZ, IMGSZ, "detect", nc, None)
    net.eval()

    im = torch.zeros(1, 3, IMGSZ, IMGSZ)
    with torch.no_grad():
        det, depth = net(im)
    print(f"PT forward : det {tuple(det.shape)}  depth {tuple(depth.shape)}  nc={nc}")

    torch2onnx(net, im, ONNX_PATH, opset=OPSET, input_names=["images"],
               output_names=["output0", "depth"], dynamic=None)
    print(f"ONNX saved : {ONNX_PATH}  ({ONNX_PATH.stat().st_size / 1e6:.1f} MB)")

    # ---- 校验输出 ----
    import onnx

    g = onnx.load(str(ONNX_PATH)).graph
    print("input      :", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in g.input])
    for o in g.output:
        print("output     :", o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
    assert len(g.output) == 2, "期望 2 个输出"
    print("OK")


if __name__ == "__main__":
    main()
