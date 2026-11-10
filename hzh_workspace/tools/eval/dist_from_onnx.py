# -*- coding: utf-8 -*-
"""用 ONNX（与 RKNN 输出逐位一致的同一张图）在 GT 框内取距离中位数，输出 CSV。

用法: dist_from_onnx.py <onnx> <images_dir> <labels_dist_dir> <stems.txt> <out.csv>
口径与板端 eval_any.dist_in_box 一致：letterbox 640 画布坐标、各层 anchor 中心落在框内、
取全部 anchor 的中位数。ONNX 输入 float NCHW，需 /255、RGB。
"""
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

onnx_path, images_dir, labels_dir, stems_txt, out_csv = sys.argv[1:6]
DIST_MAX = 15.0
STRIDES = (8, 16, 32)

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
out_names = [o.name for o in sess.get_outputs()]


def letterbox(img, size=640):
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top:top + nh, left:left + nw] = r
    return canvas, s, left, top


def dist_in_box(dmaps, box):
    x1, y1, x2, y2 = box
    vals = []
    for m, st in zip(dmaps, STRIDES):
        xs = (np.arange(m.shape[1]) + 0.5) * st
        ys = (np.arange(m.shape[0]) + 0.5) * st
        mx = (xs >= x1) & (xs <= x2)
        my = (ys >= y1) & (ys <= y2)
        if mx.any() and my.any():
            vals.append(m[np.ix_(my, mx)].reshape(-1))
    if not vals:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        m = dmaps[0]
        return float(m[min(max(int(cy / STRIDES[0]), 0), m.shape[0] - 1),
                       min(max(int(cx / STRIDES[0]), 0), m.shape[1] - 1)])
    return float(np.median(np.concatenate(vals)))


rows = ["stem cls gt_m pred_m err_m"]
n_used = 0
for stem in (l.strip() for l in open(stems_txt) if l.strip()):
    base = os.path.splitext(stem)[0]
    ip = os.path.join(images_dir, stem)
    lp = os.path.join(labels_dir, base + ".txt")
    if not (os.path.isfile(ip) and os.path.isfile(lp)):
        continue
    img = cv2.imread(ip)
    if img is None:
        continue
    h, w = img.shape[:2]
    canvas, s, left, top = letterbox(img)
    inp = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255.0
    outs = sess.run(out_names, {in_name: inp})
    dmaps = [np.asarray(outs[9 + k], np.float32).reshape(np.asarray(outs[9 + k]).shape[-2:]) for k in range(3)]
    for line in open(lp):
        t = line.split()
        if len(t) < 6:
            continue
        cls = int(float(t[0]))
        cx, cy, bw, bh = (float(v) for v in t[1:5])
        d_norm = float(t[5])
        if d_norm < 0:
            continue
        gt_m = min(max(d_norm, 0.0), 1.0) * DIST_MAX
        box = [(cx - bw / 2) * w * s + left, (cy - bh / 2) * h * s + top,
               (cx + bw / 2) * w * s + left, (cy + bh / 2) * h * s + top]
        pm = dist_in_box(dmaps, box)
        rows.append("%s %d %.3f %.3f %.3f" % (base, cls, gt_m, pm, pm - gt_m))
    n_used += 1

open(out_csv, "w").write("\n".join(rows) + "\n")
print("images", n_used, "rows", len(rows) - 1, "->", out_csv)
