#!/home/kang/miniforge3/envs/yolo/bin/python
"""Infer val own_video images with a detdist checkpoint, draw head distance.

Outputs: <out>/images/*.jpg (box + cls conf + d=X.Xm, GT in label when matched),
<out>/boxes.csv, <out>/summary.json (head vs labels_dist_cal GT).
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = "/home/kang/workspace/projects/yolo-depth/ultralytics-depth"
sys.path[:0] = [REPO + "/hzh_workspace/libs", REPO]
import metdist  # noqa: E402  legacy detdist_v2 pickle aliases

CANVAS = 640
OWN_VIDEO = re.compile(r"^video\d+(_\d+min)?_frame_\d+$")
NAMES = {0: "person", 1: "bag"}


def letterbox(im, size=CANVAS):
    h, w = im.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) // 2, (size - nh) // 2
    out = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = cv2.copyMakeBorder(out, dh, size - nh - dh, dw, size - nw - dw,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, r, dw, dh


def nms(boxes, scores, thr):
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        b = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(a + b - inter, 1e-9)
        order = rest[iou < thr]
    return np.asarray(keep, dtype=np.int64)


def read_gt(label_path):
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 6 or float(p[5]) < 0:
            continue
        c, cx, cy, w, h, dn = p[0], *map(float, p[1:6])
        out.append((int(float(c)), cx, cy, w, h, float(dn) * 15.0))
    return out


def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-9)


def head_dist(dist_b, feats, bx, pct=50.0, near_pct=0.0):
    vals, off = [], 0
    for k, f in enumerate(feats):
        hh, ww = f.shape[2:]
        st = 8 * (2 ** k)
        ys, xs = np.meshgrid((np.arange(hh) + 0.5) * st, (np.arange(ww) + 0.5) * st, indexing="ij")
        vals.append(dist_b[off:off + hh * ww][((xs >= bx[0]) & (xs <= bx[2]) &
                                               (ys >= bx[1]) & (ys <= bx[3])).ravel()])
        off += hh * ww
    v = np.concatenate(vals)
    v = v[np.isfinite(v) & (v > 0)]
    if not v.size:
        return float("nan")
    if near_pct > 0 and float(np.median(v)) < 5.0:
        return float(np.percentile(v, near_pct))
    return float(np.percentile(v, pct))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="/mnt/data2/hzh/PF_11.9/images/val")
    ap.add_argument("--labels", default="/mnt/data2/hzh/PF_11.9/labels_dist_cal/val")
    ap.add_argument("--weights", default=REPO + "/hzh_workspace/runs/detdist_pf_cal/weights/best.pt")
    ap.add_argument("--out", default="/tmp/mdist/cal_ownvideo")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--logdepth", action="store_true", help="v2 head: exp(clamp(logits)) instead of sigmoid*15")
    ap.add_argument("--pct", type=float, default=50.0, help="in-box anchor percentile (50=median, <50 reads lower)")
    ap.add_argument("--near-pct", type=float, default=0.0,
                    help="if >0: use this percentile when the median reading <5m, else --pct")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = (ck.get("ema") or ck["model"]).float().to(dev).eval()

    files = sorted(p for p in Path(args.images).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"} and OWN_VIDEO.match(p.stem))
    if args.limit:
        files = files[: args.limit]
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    rows, matched = [], []
    buf_im, buf_meta = [], []

    def flush():
        if not buf_im:
            return
        x = torch.from_numpy(np.stack([i.transpose(2, 0, 1) for i in buf_im])).float().to(dev) / 255.0
        with torch.no_grad():
            y, preds = model(x)
        o2m = preds.get("one2many", preds) if isinstance(preds, dict) else preds
        if args.logdepth:
            dist = torch.exp(o2m["dist"][:, 0, :].clamp(np.log(0.2), np.log(15.0))).cpu().numpy()
        else:
            dist = (torch.sigmoid(o2m["dist"][:, 0, :]) * 15.0).cpu().numpy()
        det = y.float().cpu().numpy()
        for b, (path, r, dw, dh, im0) in enumerate(buf_meta):
            H, W = im0.shape[:2]
            d = det[b]
            boxes = np.stack([d[0] - d[2] / 2, d[1] - d[3] / 2, d[0] + d[2] / 2, d[1] + d[3] / 2], 1)
            scores = d[4:]
            cls = scores.argmax(0)
            sc = scores.max(0)
            m = sc >= args.conf
            boxes, sc, cls = boxes[m], sc[m], cls[m]
            if sc.size:
                keep = nms(boxes, sc, 0.7)
                boxes, sc, cls = boxes[keep], sc[keep], cls[keep]
            gts = read_gt(Path(args.labels) / (path.stem + ".txt"))
            gt_boxes = [[(g[1] - g[3] / 2) * W, (g[2] - g[4] / 2) * H,
                         (g[1] + g[3] / 2) * W, (g[2] + g[4] / 2) * H] for g in gts]

            im = im0.copy()
            used = [False] * len(gts)
            for i in range(sc.size):
                bx = boxes[i]
                d_m = head_dist(dist[b], o2m["feats"], bx, args.pct, args.near_pct)
                orig = [(bx[0] - dw) / r, (bx[1] - dh) / r, (bx[2] - dw) / r, (bx[3] - dh) / r]
                best, bi = 0.5, -1
                for gi, gb in enumerate(gt_boxes):
                    if used[gi]:
                        continue
                    ov = iou_xyxy(orig, gb)
                    if ov >= best:
                        best, bi = ov, gi
                gt_m = None
                if bi >= 0:
                    used[bi] = True
                    gt_m = gts[bi][5]
                    if np.isfinite(d_m):
                        matched.append((gt_m, d_m, float(bx[3] - bx[1])))
                rows.append({
                    "image": path.name, "cls": int(cls[i]), "conf": round(float(sc[i]), 4),
                    "h_canvas": round(float(bx[3] - bx[1]), 1),
                    "pred_m": round(d_m, 3) if np.isfinite(d_m) else "",
                    "gt_cal_m": round(gt_m, 3) if gt_m is not None else "",
                    "iou_with_gt": round(best, 3) if bi >= 0 else "",
                })
                col = (0, 165, 255)
                x1, y1, x2, y2 = [int(v) for v in orig]
                cv2.rectangle(im, (x1, y1), (x2, y2), col, 3)
                label = f"{NAMES.get(int(cls[i]), '?')} {sc[i]:.2f} d={d_m:.1f}m"
                if gt_m is not None:
                    label += f" (gt {gt_m:.1f})"
                cv2.putText(im, label, (x1, max(y1 - 8, 24)), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(im, label, (x1, max(y1 - 8, 24)), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, col, 2, cv2.LINE_AA)
            cv2.imwrite(str(out / "images" / f"{path.stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        buf_im.clear()
        buf_meta.clear()

    for k, p in enumerate(files):
        im0 = cv2.imread(str(p))
        if im0 is None:
            print(f"WARN unreadable: {p}")
            continue
        im, r, dw, dh = letterbox(im0)
        buf_im.append(np.ascontiguousarray(im[:, :, ::-1]))
        buf_meta.append((p, r, dw, dh, im0))
        if len(buf_im) >= args.batch:
            flush()
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(files)}", flush=True)
    flush()

    with (out / "boxes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {"weights": args.weights, "labels": args.labels, "images": len(files),
               "detections": len(rows), "matched": len(matched)}
    if matched:
        gt = np.array([m[0] for m in matched])
        pd = np.array([m[1] for m in matched])
        err = pd - gt
        rg = np.argsort(np.argsort(gt)).astype(float)
        rp = np.argsort(np.argsort(pd)).astype(float)
        rg -= rg.mean(); rp -= rp.mean()
        rho = float((rg * rp).sum() / max(np.sqrt((rg ** 2).sum() * (rp ** 2).sum()), 1e-9))
        buckets = {}
        for lo, hi in [(0, 5), (5, 10), (10, 15)]:
            s = (gt >= lo) & (gt < hi if hi < 15 else gt <= hi)
            buckets[f"{lo}-{hi}"] = {
                "n": int(s.sum()),
                "gt_median": round(float(np.median(gt[s])), 2) if s.any() else None,
                "pred_median": round(float(np.median(pd[s])), 2) if s.any() else None,
                "mae": round(float(np.mean(np.abs(err[s]))), 3) if s.any() else None,
            }
        hh = np.array([m[2] for m in matched])
        inv = 1.0 / np.maximum(hh, 1.0)
        ri = np.argsort(np.argsort(inv)).astype(float)
        rp2 = np.argsort(np.argsort(pd)).astype(float)
        ri -= ri.mean(); rp2 -= rp2.mean()
        rho_h = float((ri * rp2).sum() / max(np.sqrt((ri ** 2).sum() * (rp2 ** 2).sum()), 1e-9))
        hbuckets = {}
        for lo, hi in [(0, 40), (40, 80), (80, 160), (160, 320), (320, 640)]:
            s = (hh >= lo) & (hh < hi)
            hbuckets[f"h{lo}-{hi}"] = {
                "n": int(s.sum()),
                "gt_median": round(float(np.median(gt[s])), 2) if s.any() else None,
                "pred_median": round(float(np.median(pd[s])), 2) if s.any() else None,
            }
        summary.update({
            "mae": round(float(np.mean(np.abs(err))), 3),
            "bias_median": round(float(np.median(err)), 3),
            "median_abs_err": round(float(np.median(np.abs(err))), 3),
            "rho": round(rho, 4),
            "rho_pred_vs_inv_height": round(rho_h, 4),
            "height_buckets": hbuckets,
            "recall_vs_gt": round(len(matched) / max(sum(len(read_gt(Path(args.labels) / (p.stem + ".txt"))) for p in files), 1), 4),
            "buckets": buckets,
        })
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()