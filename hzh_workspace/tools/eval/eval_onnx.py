# -*- coding: utf-8 -*-
"""用 onnxruntime 跑 fp32 onnx，输出与 C++ --dump 相同格式的预测（stem cls score x1 y1 x2 y2）。
字母框与 yolo-det 的 C++ 完全一致：等比缩放、宽 4 / 高 2 对齐、居中 pad 114、RGB /255。"""
import os
import sys
import glob
import numpy as np
import cv2
import onnxruntime as ort

ONNX = sys.argv[1]
IMAGES = sys.argv[2]
OUT = sys.argv[3]
CONF = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
NMS = float(sys.argv[5]) if len(sys.argv) > 5 else 0.45
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


def nms(boxes, scores, thr):
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        b = boxes[i]
        rest = order[1:]
        xx1 = np.maximum(boxes[rest, 0], b[0])
        yy1 = np.maximum(boxes[rest, 1], b[1])
        xx2 = np.minimum(boxes[rest, 2], b[2])
        yy2 = np.minimum(boxes[rest, 3], b[3])
        iw = np.maximum(0, xx2 - xx1)
        ih = np.maximum(0, yy2 - yy1)
        inter = iw * ih
        uni = (b[2] - b[0]) * (b[3] - b[1]) + \
              (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1]) - inter
        iou = np.where(uni > 0, inter / np.maximum(uni, 1e-9), 0)
        order = rest[iou <= thr]
    return np.array(keep, dtype=int)


def main():
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    lines = []
    for p in sorted(glob.glob(os.path.join(IMAGES, "*"))):
        if not p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        img = cv2.imread(p)
        stem = os.path.splitext(os.path.basename(p))[0]
        canvas, s, xp, yp = letterbox(img)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None]
        out = sess.run(None, {iname: blob})[0][0]  # (5,8400)
        d = out.T
        sc = d[:, 4]
        m = sc >= CONF
        d, sc = d[m], sc[m]
        if len(d):
            cx, cy, w, h = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
            boxes = np.stack([(cx - w / 2) * W, (cy - h / 2) * H,
                              (cx + w / 2) * W, (cy + h / 2) * H], axis=1)
            keep = nms(boxes, sc, NMS)
            for b, sc_ in zip(boxes[keep], sc[keep]):
                x1 = (b[0] - xp) / s
                y1 = (b[1] - yp) / s
                x2 = (b[2] - xp) / s
                y2 = (b[3] - yp) / s
                ih, iw = img.shape[:2]
                x1 = min(max(x1, 0), iw - 1); x2 = min(max(x2, 0), iw - 1)
                y1 = min(max(y1, 0), ih - 1); y2 = min(max(y2, 0), ih - 1)
                lines.append("%s 0 %.6f %.2f %.2f %.2f %.2f" % (stem, sc_, x1, y1, x2, y2))
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("preds: %d -> %s" % (len(lines), OUT))


if __name__ == "__main__":
    main()
