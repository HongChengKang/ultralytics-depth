# -*- coding: utf-8 -*-
"""用 ONNX 同时算两种距离头的「框内中位距离」。

- 10 输出（旧 pf_v2）：outs[9] 是稠密深度图 (1,1,S,S)，单位米；在框内取有效像素中位。
- 12 输出（新 dist_pf_all）：outs[9..11] 是 P3/P4/P5 的逐 anchor 距离；取框内 anchor 中位。

用法:
  dist_both.py <onnx> <images_dir> <labels_dist_dir> <stems.txt> <out_gt.csv>
  dist_both.py <onnx> <images_dir> <labels_dist_dir> <stems.txt> <out.csv> --preds <preds7.txt> <out_dist.txt>
"""
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

args = sys.argv[1:]
preds_in = preds_out = None
if "--preds" in args:
    i = args.index("--preds")
    preds_in, preds_out = args[i + 1], args[i + 2]
    args = args[:i]

onnx_path, images_dir, labels_dir, stems_txt, out_csv = args[:5]
DIST_MAX = 15.0
STRIDES = (8, 16, 32)

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
out_names = [o.name for o in sess.get_outputs()]
N_OUT = len(out_names)


def letterbox(img, size=640):
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top:top + nh, left:left + nw] = r
    return canvas, s, left, top


def med_anchor(dmaps, box):
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


def med_dense(dmap, box, min_px=4):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(dmap.shape[1], max(x1 + 1, x2)), min(dmap.shape[0], max(y1 + 1, y2))
    patch = dmap[y1:y2, x1:x2].reshape(-1)
    patch = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(patch)) if patch.size >= min_px else float("nan")


pred_map = {}
if preds_in:
    for line in open(preds_in):
        t = line.split()
        if len(t) >= 7:
            pred_map.setdefault(t[0], []).append(
                (int(float(t[1])), float(t[2]), [float(v) for v in t[3:7]]))

gt_rows = ["stem cls gt_m pred_m err_m"]
dist_rows = []
n_img = 0
for stem in (l.strip() for l in open(stems_txt) if l.strip()):
    base = os.path.splitext(stem)[0]
    ip, lp = os.path.join(images_dir, stem), os.path.join(labels_dir, base + ".txt")
    if not (os.path.isfile(ip) and os.path.isfile(lp)):
        continue
    img = cv2.imread(ip)
    if img is None:
        continue
    h, w = img.shape[:2]
    canvas, s, left, top = letterbox(img)
    inp = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255.0
    outs = sess.run(out_names, {in_name: inp})

    if N_OUT >= 12:
        dmaps = [np.asarray(outs[9 + k], np.float32).reshape(np.asarray(outs[9 + k]).shape[-2:])
                 for k in range(3)]
        f = med_anchor
    else:
        dmaps = np.asarray(outs[9], np.float32).reshape(np.asarray(outs[9]).shape[-2:])
        f = med_dense

    for line in open(lp):
        t = line.split()
        if len(t) < 6:
            continue
        d_norm = float(t[5])
        if d_norm < 0:
            continue
        cx, cy, bw, bh = (float(v) for v in t[1:5])
        gt_m = min(max(d_norm, 0.0), 1.0) * DIST_MAX
        box = [(cx - bw / 2) * w * s + left, (cy - bh / 2) * h * s + top,
               (cx + bw / 2) * w * s + left, (cy + bh / 2) * h * s + top]
        pm = f(dmaps, box)
        gt_rows.append("%s %d %.3f %.3f %.3f" % (base, int(float(t[0])), gt_m, pm, pm - gt_m))

    if preds_in:
        for cls, sc, (x1, y1, x2, y2) in pred_map.get(base, []):
            box = [x1 * s + left, y1 * s + top, x2 * s + left, y2 * s + top]
            pm = f(dmaps, box)
            dist_rows.append("%s %d %.6f %.2f %.2f %.2f %.2f %.3f"
                             % (base, cls, sc, x1, y1, x2, y2, pm))
    n_img += 1

open(out_csv, "w").write("\n".join(gt_rows) + "\n")
if preds_out:
    open(preds_out, "w").write("\n".join(dist_rows) + "\n")
print("out=%d images=%d gt_rows=%d dist_rows=%d" % (N_OUT, n_img, len(gt_rows) - 1, len(dist_rows)))
