# -*- coding: utf-8 -*-
"""用 ultralytics 的 best.pt (YOLO26-DetDepth) 跑 PT 推理，预处理与 yolo-det 的 C++ 评估完全一致：
   先按 C++ 的 letterbox 规则生成 640x640 画布(114 pad) → 送 PT → 框再逆变换回原图 →
   输出与 C++ `--dump` 同格式的预测文件: stem cls score x1 y1 x2 y2

用法:
  python pt_predict.py <model.pt> <images_dir> <out.txt> [--conf 0.25] [--iou 0.45] [--device 0]
"""
import os, sys, glob
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ["DEPTH_REPO"]).resolve()
sys.path[:0] = [str(REPO / "hck_workspace" / "detdepth"), str(REPO)]

import cv2
import numpy as np

W = H = 640


def letterbox(img):
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    rw, rh = int(w * s), int(h * s)
    rw -= rw % 4
    rh -= rh % 2
    xp = (W - rw) // 2
    xp -= xp % 4
    yp = (H - rh) // 2
    yp -= yp % 2
    res = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((H, W, 3), 114, dtype=np.uint8)
    canvas[yp:yp + rh, xp:xp + rw] = res
    return canvas, s, xp, yp


def main():
    args = sys.argv[1:]
    conf, iou, device = 0.25, 0.45, "0"
    pos, i = [], 0
    while i < len(args):
        a = args[i]
        if a == "--conf":
            conf = float(args[i + 1]); i += 2
        elif a == "--iou":
            iou = float(args[i + 1]); i += 2
        elif a == "--device":
            device = args[i + 1]; i += 2
        else:
            pos.append(a); i += 1
    model_path, images_dir, out_path = pos[0], pos[1], pos[2]

    from detdepth import DetDepthPredictor  # noqa: E402
    from ultralytics import YOLO  # noqa: E402

    model = YOLO(model_path)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    paths = sorted([p for p in glob.glob(os.path.join(images_dir, "*")) if os.path.splitext(p)[1].lower() in exts])
    lines = []
    for p in paths:
        img = cv2.imread(p)
        oh, ow = img.shape[:2]
        canvas, s, xp, yp = letterbox(img)
        r = model.predict(canvas, predictor=DetDepthPredictor, imgsz=640, conf=conf, iou=iou,
                          verbose=False, device=device)[0]
        n = 0
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            sc = r.boxes.conf.cpu().numpy()
            cl = r.boxes.cls.cpu().numpy().astype(int)
            stem = os.path.splitext(os.path.basename(p))[0]
            for b, s_, c in zip(xyxy, sc, cl):
                x1 = min(max((float(b[0]) - xp) / s, 0.0), ow - 1)
                y1 = min(max((float(b[1]) - yp) / s, 0.0), oh - 1)
                x2 = min(max((float(b[2]) - xp) / s, 0.0), ow - 1)
                y2 = min(max((float(b[3]) - yp) / s, 0.0), oh - 1)
                lines.append("%s %d %.6f %.2f %.2f %.2f %.2f" % (stem, c, s_, x1, y1, x2, y2))
                n += 1
        print("[pt] %-40s preds=%d" % (os.path.basename(p), n))
    with open(out_path, "w") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
    print("[pt] %d images, %d boxes -> %s" % (len(paths), len(lines), out_path))


if __name__ == "__main__":
    main()
