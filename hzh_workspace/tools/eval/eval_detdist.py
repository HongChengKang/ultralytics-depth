#!/home/kang/miniforge3/envs/yolo/bin/python
"""detdist evaluation: mAP from the run + distance consistency on labelled boxes.

Reported distance metrics are *consistency with the aggregated teacher labels*
(``labels_dist``), NOT ground-truth metric accuracy: the labels themselves are a
per-box median of a distilled monocular depth model.

  * mAP50 / mAP50-95 : copied from runs/<name>/results.csv (best or last epoch)
  * per-bucket 0-5 / 5-10 / 10-15 m : n boxes, median predicted d_m, median label
    d_m, median signed error, MAE
  * Spearman rho (numpy ranks) between predicted d_m and label d_m over all
    boxes with a valid distance
  * boxes with d_norm == -1 are counted but excluded from distance metrics
    (their detection contribution already sits in mAP)

Usage:
  tools/eval/eval_detdist.py --weights runs/dist_pf_all/weights/best.pt
  tools/eval/eval_detdist.py --run runs/detdist_pf --baseline runs/detdist_baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve()
WS = HERE.parents[2]                     # .../hzh_workspace
REPO = HERE.parents[3]                   # .../ultralytics-depth
sys.path[:0] = [str(WS / "libs"), str(REPO)]

from detdist.data import DistDataset  # noqa: E402

DIST_MAX = 15.0
BUCKETS = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho without scipy."""
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def read_map(run_dir: Path) -> dict:
    """Return the last and best rows of results.csv (column names kept verbatim)."""
    csv_path = run_dir / "results.csv"
    if not csv_path.is_file():
        return {"error": f"missing {csv_path}"}
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"error": f"empty {csv_path}"}
    keys = [k.strip() for k in rows[0].keys()]
    clean = [{k.strip(): (float(v) if k.strip() not in {"epoch", "time"} or v.strip().replace(".", "", 1).isdigit() else v) for k, v in r.items()} for r in rows]

    def pick(key):
        vals = [(int(float(r[key])), r) for r in clean if r.get(key) not in (None, "")]
        if not vals:
            return None
        return max(vals, key=lambda t: t[0])[1]

    m50 = "metrics/mAP50(B)"
    m5095 = "metrics/mAP50-95(B)"
    best = max(clean, key=lambda r: float(r.get(m5095, -1)))
    last = clean[-1]
    return {
        "csv": str(csv_path),
        "epochs_logged": len(clean),
        "columns": keys,
        "best": {k: best.get(k) for k in ("epoch", m50, m5095, "metrics/precision(B)", "metrics/recall(B)")},
        "last": {k: last.get(k) for k in ("epoch", m50, m5095, "metrics/precision(B)", "metrics/recall(B)")},
    }


@torch.no_grad()
def collect(weights: Path, data_yaml: Path, labels_root: Path, imgsz, device: str, batch: int, workers: int):
    cfg = yaml.safe_load(data_yaml.read_text())
    data = {"names": cfg["names"], "nc": len(cfg["names"]), "channels": 3}
    ds = DistDataset(
        img_path=str(Path(cfg["path"]) / cfg["val"]),
        imgsz=tuple(imgsz),
        batch_size=batch,
        augment=False,
        hyp=type("H", (), {"mask_ratio": 4, "overlap_mask": True, "bgr": 0.0})(),
        rect=False,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.5,
        prefix="val: ",
        task="detect",
        classes=None,
        data=data,
        fraction=1.0,
        labels_root=str(labels_root),
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=DistDataset.collate_fn
    )

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model = ckpt.get("ema") or ckpt["model"]
    model = model.float().to(device).eval()

    gt_all, pred_all, cls_all = [], [], []
    for batch in loader:
        img = batch["img"].to(device).float() / 255.0
        _, preds = model(img)
        preds = preds["one2many"] if "one2many" in preds else preds
        dist = torch.sigmoid(preds["dist"][:, 0, :]).float().cpu().numpy() * DIST_MAX  # (b, A_all)
        # anchor grid of every level, in the same order the head concatenates them (P3 -> P5)
        levels, off = [], 0
        for k, feat in enumerate(preds["feats"]):
            hh, ww = feat.shape[2:]
            levels.append((off, hh, ww, 8 * (2**k)))
            off += hh * ww

        n_anchor = sum(hh * ww for _o, hh, ww, _s in levels)
        if dist.shape[1] != n_anchor:
            raise RuntimeError(
                f"dist head has {dist.shape[1]} anchors but the feature maps give {n_anchor}: "
                "the checkpoint was trained with a different distance head (e.g. the old P3-only build)"
            )

        bboxes = batch["bboxes"].numpy()  # normalized xywh in the letterboxed input
        dist_gt = batch["dist"].numpy().reshape(-1)
        bidx = batch["batch_idx"].numpy().reshape(-1)
        H, W = img.shape[2:]
        for i in range(bboxes.shape[0]):
            b = int(bidx[i])
            cx, cy, bw, bh = bboxes[i]
            x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
            x2, y2 = (cx + bw / 2) * W, (cy + bh / 2) * H
            vals = []
            for off_l, hh, ww, st in levels:
                xs = (np.arange(ww) + 0.5) * st
                ys = (np.arange(hh) + 0.5) * st
                xs_in = np.nonzero((xs >= x1) & (xs <= x2))[0]
                ys_in = np.nonzero((ys >= y1) & (ys <= y2))[0]
                if xs_in.size and ys_in.size:
                    gx, gy = np.meshgrid(xs_in, ys_in)
                    vals.append(dist[b][off_l + (gy * ww + gx).ravel()])
            if not vals:  # degenerate/thin box: use the finest-level anchor under the centre
                off_l, hh, ww, st = levels[0]
                cxi = min(max(int(cx * W) // st, 0), ww - 1)
                cyi = min(max(int(cy * H) // st, 0), hh - 1)
                vals = [dist[b][off_l + cxi + cyi * ww : off_l + cxi + cyi * ww + 1]]
            vals = np.concatenate(vals)
            gt_all.append(dist_gt[i])
            pred_all.append(float(np.median(vals)))
            cls_all.append(1 if dist_gt[i] >= 0 else 0)
    return np.asarray(gt_all, np.float64), np.asarray(pred_all, np.float64), ds


def summarize(gt_norm: np.ndarray, pred_m: np.ndarray) -> dict:
    valid = gt_norm >= 0
    gt_m = np.clip(gt_norm, 0, 1) * DIST_MAX
    out = {
        "boxes_total": int(gt_norm.size),
        "boxes_valid": int(valid.sum()),
        "boxes_ignore": int((~valid).sum()),
        "spearman_rho": spearman(gt_m[valid], pred_m[valid]),
        "mae_m": float(np.mean(np.abs(pred_m[valid] - gt_m[valid]))) if valid.any() else None,
        "median_abs_err_m": float(np.median(np.abs(pred_m[valid] - gt_m[valid]))) if valid.any() else None,
        "buckets": {},
    }
    for lo, hi in BUCKETS:
        m = valid & (gt_m >= lo) & (gt_m < hi if hi < DIST_MAX else gt_m <= hi)
        if not m.any():
            out["buckets"][f"{lo:g}-{hi:g}"] = {"n": 0}
            continue
        out["buckets"][f"{lo:g}-{hi:g}"] = {
            "n": int(m.sum()),
            "gt_median_m": float(np.median(gt_m[m])),
            "pred_median_m": float(np.median(pred_m[m])),
            "signed_bias_m": float(np.median(pred_m[m] - gt_m[m])),
            "mae_m": float(np.mean(np.abs(pred_m[m] - gt_m[m]))),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="checkpoint, default <run>/weights/best.pt")
    ap.add_argument("--run", default=str(WS / "runs" / "detdist_pf"), help="run dir for results.csv")
    ap.add_argument("--data", default=str(WS / "configs" / "detdist_data.yaml"))
    ap.add_argument("--labels-root", default=None, help="default: <dataset_root>/labels_dist")
    ap.add_argument("--imgsz", default="640", help="eval letterbox size H,W (match training)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--baseline", default=None, help="baseline run dir whose results.csv is compared")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    weights = Path(args.weights) if args.weights else run_dir / "weights" / "best.pt"
    if args.labels_root is None:
        ds_cfg = WS / "configs" / "detdist_dataset.yaml"
        root = "/mnt/data2/hzh/PF_11.9"
        if ds_cfg.is_file():
            root = (yaml.safe_load(ds_cfg.read_text()) or {}).get("dataset_root", root)
        args.labels_root = str(Path(root) / "labels_dist")
    sz = tuple(int(v) for v in str(args.imgsz).replace("x", ",").split(","))
    imgsz = (sz[0], sz[-1])
    device = str(args.device)
    if device.isdigit():
        device = f"cuda:{device}"
    elif device in {"", "auto"}:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gt, pred, ds = collect(weights, Path(args.data), Path(args.labels_root), imgsz, device, args.batch, args.workers)

    report = {
        "weights": str(weights),
        "data": str(args.data),
        "labels_root": args.labels_root,
        "imgsz": list(imgsz),
        "note": "distance metrics = consistency with aggregated teacher labels, not GT metric accuracy",
        "dataset": {"images": len(ds.labels), "label_stats": ds.dist_stats},
        "map": read_map(run_dir),
        "distance": summarize(gt, pred),
    }
    if args.baseline:
        report["baseline_map"] = read_map(Path(args.baseline))
    out = Path(args.out) if args.out else run_dir / "dist_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    d = report["distance"]
    print(f"\n=== detdist eval: {weights} ===")
    print(f"boxes: total={d['boxes_total']} valid={d['boxes_valid']} ignore={d['boxes_ignore']}")
    print(f"Spearman rho={d['spearman_rho']:.4f}  MAE={d['mae_m']:.3f} m  median|err|={d['median_abs_err_m']:.3f} m")
    print(f"{'bucket':>8} {'n':>6} {'gt_med':>8} {'pred_med':>9} {'bias':>8} {'mae':>7}")
    for k, v in d["buckets"].items():
        if v["n"] == 0:
            print(f"{k:>8} {0:>6}")
            continue
        print(f"{k:>8} {v['n']:>6} {v['gt_median_m']:>8.3f} {v['pred_median_m']:>9.3f} {v['signed_bias_m']:>8.3f} {v['mae_m']:>7.3f}")
    for tag in ("best", "last"):
        m = report["map"].get(tag)
        if m:
            print(f"mAP[{tag}] epoch={m['epoch']} mAP50={m['metrics/mAP50(B)']:.4f} mAP50-95={m['metrics/mAP50-95(B)']:.4f}")
    if "baseline_map" in report and "best" in report["baseline_map"]:
        b = report["baseline_map"]["best"]
        m = report["map"].get("best")
        if m:
            print(
                f"baseline mAP50={b['metrics/mAP50(B)']:.4f} mAP50-95={b['metrics/mAP50-95(B)']:.4f} "
                f"-> delta mAP50={m['metrics/mAP50(B)'] - b['metrics/mAP50(B)']:+.4f} "
                f"delta mAP50-95={m['metrics/mAP50-95(B)'] - b['metrics/mAP50-95(B)']:+.4f}"
            )
    print(f"report: {out}")


if __name__ == "__main__":
    main()
