# -*- coding: utf-8 -*-
"""距离 + 检测可视化：一张图一个模型 = 两色框（黄=GT，红=预测），框上都写距离(m)。

图上方黑条：文件名 | GT 框数 / 预测框数 | 匹配框(IoU>=0.5) 的距离 MAE / 中位偏差

用法:
  python3 vis_dist3.py <images> <labels_dist> <preds.dist.txt> <out_dir> [--stems FILE]
      [--title 模型名] [--max 0]
"""
import argparse
import glob
import os

import cv2
import numpy as np

DIST_MAX = 15.0
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
C_GT = (0, 255, 255)      # 黄
C_PRED = (0, 0, 255)      # 红


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_preds(path):
    out = {}
    if not os.path.isfile(path):
        return out
    for line in open(path):
        t = line.split()
        if len(t) < 7:
            continue
        dm = float(t[7]) if len(t) >= 8 else None
        out.setdefault(t[0], []).append(
            (int(t[1]), float(t[2]), np.array([float(t[3]), float(t[4]), float(t[5]), float(t[6])]), dm))
    return out


def tag(img, x, y, text, color, scale=0.42, th=1):
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    y = max(y, tht + 2)
    cv2.rectangle(img, (x, y - tht - 2), (x + tw + 2, y + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 1, y - 1), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("labels_dist")
    ap.add_argument("preds")
    ap.add_argument("out")
    ap.add_argument("--stems", default="")
    ap.add_argument("--title", default="")
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    preds = load_preds(args.preds)
    files = sorted(f for f in glob.glob(os.path.join(args.images, "*"))
                   if os.path.splitext(f)[1].lower() in IMG_EXTS)
    if args.stems:
        keep = {os.path.splitext(l.strip())[0] for l in open(args.stems) if l.strip()}
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in keep]
    if args.max:
        files = files[:args.max]
    os.makedirs(args.out, exist_ok=True)

    n_img = 0
    all_errs = []
    for f in files:
        errs = []
        stem = os.path.splitext(os.path.basename(f))[0]
        img = cv2.imread(f)
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = []
        lab = os.path.join(args.labels_dist, stem + ".txt")
        if os.path.isfile(lab):
            for line in open(lab):
                t = line.split()
                if len(t) < 6:
                    continue
                cx, cy, bw, bh = (float(v) for v in t[1:5])
                d_norm = float(t[5])
                box = np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
                gts.append((box, d_norm * DIST_MAX if d_norm >= 0 else None))

        for box, dm in gts:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(img, (x1, y1), (x2, y2), C_GT, 2)
            if dm is not None and (y2 - y1) >= 18:
                tag(img, x1, y1, "G %.1fm" % dm, C_GT)

        for cls, sc, box, dm in preds.get(stem, []):
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(img, (x1, y1), (x2, y2), C_PRED, 2)
            if dm is not None and (y2 - y1) >= 18:
                tag(img, x1, min(h - 2, y2 + 16), "P %.1fm" % dm, C_PRED)
            best, bi = 0.0, -1
            for i, (gb, _gd) in enumerate(gts):
                v = iou(box, gb)
                if v > best:
                    best, bi = v, i
            if bi >= 0 and best >= 0.5 and gts[bi][1] is not None and dm is not None:
                errs.append(dm - gts[bi][1])

        bar = np.zeros((26, img.shape[1], 3), np.uint8)
        txt = "%s  GT=%d  PRED=%d" % (stem, len(gts), len(preds.get(stem, [])))
        if errs:
            txt += "   MAE=%.1fm bias=%+.1fm n=%d" % (np.abs(errs).mean(), np.median(errs), len(errs))
            all_errs.extend(errs)
        if args.title:
            txt = args.title + "  |  " + txt
        cv2.putText(bar, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if img.shape[1] >= 860:
            cv2.putText(bar, "GT", (img.shape[1] - 96, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_GT, 1, cv2.LINE_AA)
            cv2.putText(bar, "PRED", (img.shape[1] - 52, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_PRED, 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.out, stem + ".jpg"), np.vstack([bar, img]), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        n_img += 1

    suffix = ("  total matched MAE=%.2fm bias=%.2fm n=%d"
              % (np.abs(all_errs).mean(), np.median(all_errs), len(all_errs))) if all_errs else ""
    print("wrote %d images to %s%s" % (n_img, args.out, suffix))


if __name__ == "__main__":
    main()
