#!/home/kang/miniforge3/envs/yolo/bin/python
"""One-step distance labels: teacher depth model -> 6-column yolo labels.

For every image of a split this script runs the depth teacher, aggregates the
predicted depth over each *existing* detection box and writes

    <output_dir>/<split>/<stem>.txt      class cx cy w h d_norm

Nothing is cached on disk: no ``.npy`` depth maps, no dense depth files.  The
first five columns are copied byte-for-byte from the original label line; only
the sixth column is appended.

d_norm rules (identical to the previous two-stage pipeline)
----------------------------------------------------------
* pixel box = floor((c-w/2)*W), ceil((c+w/2)*W) clipped to the image
* valid pixels = finite depth with ``0 < d <= max_dist``
* at least ``min_valid_pixels`` of them -> d_m = median, d_norm = round(d_m/max_dist, 6)
* otherwise (or a box whose raw median already exceeds max_dist) -> d_norm = -1
  (distance ignored, detection label kept)

Usage
-----
  tools/data/make_dist_labels.py --limit 20 --output-dir /tmp/labels_dist_smoke
  tools/data/make_dist_labels.py                 # train+val, resume by default
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

HERE = Path(__file__).resolve()
WS = HERE.parents[2]                      # .../hzh_workspace
REPO = HERE.parents[3]                    # .../ultralytics-depth
sys.path[:0] = [str(REPO)]

DEFAULT_ROOT = "/mnt/data2/hzh/PF_11.9"
DEFAULT_WEIGHTS = str(REPO / "weights" / "yolo26x-depth.pt")
DATASET_CFG = WS / "configs" / "detdist_dataset.yaml"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TOKEN_RE = re.compile(r"\S+")
SANITY_IMAGES = 8


# --------------------------------------------------------------------------- io
def read_label_lines(path: Path) -> tuple[list[str], list[list[str]]]:
    """Return raw lines and their whitespace tokens (5 columns each)."""
    text = path.read_text(encoding="utf-8")
    raw = text.splitlines()
    lines, toks = [], []
    for ln in raw:
        if not ln.strip():
            continue
        t = TOKEN_RE.findall(ln)
        if len(t) != 5:
            raise ValueError(f"{path}: expected 5 columns 'cls cx cy w h', got {len(t)}: {ln!r}")
        lines.append(ln)
        toks.append(t)
    return lines, toks


def line_prefix(line: str) -> str:
    """Everything up to the end of the 5th token, verbatim."""
    spans = [m.span() for m in TOKEN_RE.finditer(line)]
    return line[: spans[4][1]]


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def iter_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)


def rel_stem(img: Path, images_dir: Path) -> Path:
    return img.relative_to(images_dir).with_suffix("")


# ------------------------------------------------------------------- aggregation
def box_stats(depth: np.ndarray, tok: list[str], max_dist: float, min_pixels: int) -> dict[str, Any]:
    h, w = depth.shape[:2]
    cls = int(float(tok[0]))
    cx, cy, bw, bh = (float(v) for v in tok[1:5])
    x1 = max(0, min(w - 1, math.floor((cx - bw / 2) * w)))
    x2 = max(0, min(w - 1, math.ceil((cx + bw / 2) * w)))
    y1 = max(0, min(h - 1, math.floor((cy - bh / 2) * h)))
    y2 = max(0, min(h - 1, math.ceil((cy + bh / 2) * h)))
    out = {
        "cls": cls,
        "d_m": None,
        "d_norm": -1.0,
        "valid": 0,
        "valid_pixels": 0,
        "box_area": max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1),
        "reason": "",
    }
    if bw <= 0 or bh <= 0 or x2 <= x1 or y2 <= y1:
        out["reason"] = "degenerate_box"
        return out
    patch = depth[y1 : y2 + 1, x1 : x2 + 1]
    pos = patch[np.isfinite(patch) & (patch > 0)]
    if pos.size == 0:
        out["reason"] = "insufficient_valid_pixels"
        return out
    raw_med = float(np.median(pos))
    valid = pos[pos <= max_dist]
    out["valid_pixels"] = int(valid.size)
    if valid.size < min_pixels:
        # attribute the ignore: the box lies (mostly) beyond max_dist, or it is just too small
        out["reason"] = "median_gt_max_dist" if raw_med > max_dist else "insufficient_valid_pixels"
        out["raw_median"] = raw_med
        return out
    d_m = float(np.median(valid))
    out["d_m"] = d_m
    out["d_norm"] = round(min(max(d_m, 0.0), max_dist) / max_dist, 6)
    out["valid"] = 1
    return out


# ------------------------------------------------------------------------ stats
class SplitStats:
    def __init__(self, split: str, max_dist: float):
        self.split = split
        self.max_dist = max_dist
        self.images = 0
        self.with_labels = 0
        self.empty = 0
        self.boxes = 0
        self.valid = 0
        self.ignore = 0
        self.reasons = {"degenerate_box": 0, "median_gt_max_dist": 0, "insufficient_valid_pixels": 0}
        self.rows: list[dict[str, Any]] = []
        self.depth_medians: list[float] = []
        self.depth_max: list[float] = []

    def add(self, rel: str, results: list[dict[str, Any]], depth: np.ndarray) -> None:
        self.images += 1
        self.depth_medians.append(float(np.median(depth)))
        self.depth_max.append(float(np.max(depth)))
        if not results:
            self.empty += 1
            return
        self.with_labels += 1
        for i, r in enumerate(results):
            self.boxes += 1
            if r["valid"]:
                self.valid += 1
            else:
                self.ignore += 1
                if r["reason"]:
                    self.reasons[r["reason"]] = self.reasons.get(r["reason"], 0) + 1
            self.rows.append({"image": rel, "box_idx": i, **r})

    def summary(self, elapsed: float) -> dict[str, Any]:
        valid = [r["d_m"] for r in self.rows if r["valid"]]
        buckets: dict[str, Any] = {}
        names = []
        for lo, hi in [(0, 5), (5, 10), (10, 15)]:
            if lo >= self.max_dist:
                continue
            names.append(f"{lo:g}-{hi:g}")
        for name in names:
            lo, hi = (float(v) for v in name.split("-"))
            sel = [d for d in valid if lo <= d < (hi if hi < self.max_dist else hi + 1e-9)]
            buckets[name] = {
                "boxes": len(sel),
                "share_of_valid": (len(sel) / len(valid)) if valid else 0.0,
                "d_m_min": min(sel) if sel else None,
                "d_m_median": statistics.median(sel) if sel else None,
                "d_m_max": max(sel) if sel else None,
            }
        reasons = {k: v for k, v in self.reasons.items() if v}
        return {
            "split": self.split,
            "images": self.images,
            "images_with_labels": self.with_labels,
            "images_empty_labels": self.empty,
            "boxes_total": self.boxes,
            "boxes_valid": self.valid,
            "boxes_ignore": self.ignore,
            "valid_ratio": (self.valid / self.boxes) if self.boxes else 0.0,
            "ignore_reasons": reasons,
            "ignore_reasons_sum": sum(reasons.values()),
            "ignore_reasons_sum_ok": sum(reasons.values()) == self.ignore,
            "buckets": buckets,
            "d_m_median_valid": statistics.median(valid) if valid else None,
            "d_m_min_valid": min(valid) if valid else None,
            "d_m_max_valid": max(valid) if valid else None,
            "depth_sanity": {
                "per_image_median_min": min(self.depth_medians) if self.depth_medians else None,
                "per_image_median_median": statistics.median(self.depth_medians) if self.depth_medians else None,
                "per_image_median_max": max(self.depth_medians) if self.depth_medians else None,
                "per_image_max_max": max(self.depth_max) if self.depth_max else None,
            },
            "elapsed_sec": round(elapsed, 1),
        }


def write_csv(path: Path, rows: list[dict[str, Any]], max_dist: float) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "box_idx", "cls", "d_m", "d_norm", "valid", "valid_pixels", "box_area", "reason"])
        for r in rows:
            wr.writerow(
                [
                    r["image"],
                    r["box_idx"],
                    r["cls"],
                    "" if r["d_m"] is None else f"{r['d_m']:.4f}",
                    f"{r['d_norm']:.6f}",
                    r["valid"],
                    r["valid_pixels"],
                    r["box_area"],
                    r["reason"],
                ]
            )


# --------------------------------------------------------------------- teacher
def norm_device(device: str) -> str:
    """'0' -> 'cuda:0' (torch does not accept bare indices)."""
    d = str(device)
    if d.isdigit():
        return f"cuda:{d}"
    if d in {"", "auto"}:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return d


def load_teacher(weights: str, device: str):
    from ultralytics import YOLO

    model = YOLO(weights)
    model.to(norm_device(device))
    return model


def predict_depth(model, paths: list[Path], imgsz: int, device: str) -> list[np.ndarray]:
    from PIL import Image

    out: list[np.ndarray] = []
    results = model.predict(source=[str(p) for p in paths], imgsz=imgsz, device=norm_device(device), verbose=False)
    for r in results:
        depth = r.depth.data
        depth = depth.detach().cpu().numpy() if hasattr(depth, "detach") else np.asarray(depth)
        depth = np.asarray(depth, dtype=np.float32).squeeze()
        if depth.ndim != 2:
            raise RuntimeError(f"teacher depth has shape {depth.shape}, expected (H, W)")
        h, w = r.orig_shape
        if depth.shape != (h, w):
            import cv2

            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        out.append(depth)
    return out


# ------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-config", default=str(DATASET_CFG), help="configs/detdist_dataset.yaml")
    ap.add_argument("--dataset-root", default=None, help="overrides dataset_root from the config")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--images-dir", default=None, help="default: <root>/images/<split>")
    ap.add_argument("--labels-dir", default=None, help="default: <root>/labels/<split>")
    ap.add_argument("--output-dir", default=None, help="default: <root>/labels_dist")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="depth teacher .pt")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-dist", type=float, default=15.0)
    ap.add_argument("--min-valid-pixels", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true", help="recompute existing output labels")
    ap.add_argument("--limit", type=int, default=0, help="0 = all images, else first N per split")
    ap.add_argument("--stats-dir", default=str(WS / "stats" / "detdist"))
    ap.add_argument("--log-dir", default=str(WS / "logs"))
    ap.add_argument("--save-debug", action="store_true", help="write per-box debug json to <log-dir>/debug")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.dataset_config).read_text()) or {}
    root = Path(args.dataset_root or cfg.get("dataset_root") or DEFAULT_ROOT)
    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"teacher weights not found: {weights}")

    import hashlib

    md5 = hashlib.md5(weights.read_bytes()).hexdigest()
    model = load_teacher(str(weights), args.device)
    print(f"[dist] teacher={weights} md5={md5} imgsz={args.imgsz} max_dist={args.max_dist} "
          f"min_valid_pixels={args.min_valid_pixels}", flush=True)

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        images_dir = Path(args.images_dir) if args.images_dir else root / "images" / split
        labels_dir = Path(args.labels_dir) if args.labels_dir else root / "labels" / split
        out_dir = Path(args.output_dir) if args.output_dir else root / "labels_dist"
        out_split = out_dir / split
        if not images_dir.is_dir():
            raise SystemExit(f"images_dir not found: {images_dir}")

        images = iter_images(images_dir)
        if args.limit:
            images = images[: args.limit]
        stats = SplitStats(split, args.max_dist)
        failed: list[str] = []
        t0 = time.time()
        done = skipped = 0

        for start in range(0, len(images), args.batch):
            chunk = images[start : start + args.batch]
            todo: list[Path] = []
            for img in chunk:
                dst = out_split / rel_stem(img, images_dir).with_suffix(".txt")
                if dst.is_file() and not args.overwrite:
                    skipped += 1
                    continue
                todo.append(img)
            if not todo:
                continue
            try:
                depths = predict_depth(model, todo, args.imgsz, args.device)
            except Exception as e:  # noqa: BLE001 - one bad image must not kill the run
                print(f"[dist] batch failed ({len(todo)} imgs): {e}", flush=True)
                depths = []
                for img in todo:
                    try:
                        depths.append(predict_depth(model, [img], args.imgsz, args.device)[0])
                    except Exception as e2:  # noqa: BLE001
                        failed.append(f"{img}\t{e2}")
                        depths.append(None)

            for img, depth in zip(todo, depths):
                if depth is None:
                    continue
                rel = str(rel_stem(img, images_dir))
                lab = labels_dir / rel_stem(img, images_dir).with_suffix(".txt")
                try:
                    if lab.is_file():
                        lines, toks = read_label_lines(lab)
                    else:
                        lines, toks = [], []
                    results = [box_stats(depth, t, args.max_dist, args.min_valid_pixels) for t in toks]
                    body = "".join(f"{line_prefix(ln)} {r['d_norm']:.6f}\n" for ln, r in zip(lines, results))
                    atomic_write(out_split / rel_stem(img, images_dir).with_suffix(".txt"), body)
                    stats.add(rel, results, depth)
                    if stats.images <= SANITY_IMAGES:
                        med = float(np.median(depth))
                        print(
                            f"[sanity] {split}/{rel}: depth min={depth.min():.3f} med={med:.3f} max={depth.max():.3f} "
                            f"gt0={(depth > 0).mean():.3f} gt{args.max_dist:g}={(depth > args.max_dist).mean():.3f}",
                            flush=True,
                        )
                        if not (0.1 <= med <= 200.0):
                            print(f"[sanity][WARN] {split}/{rel}: median {med:.3f} is outside 0.1..200 — "
                                  f"teacher output may not be metric depth, confirm before training", flush=True)
                    if args.save_debug:
                        dbg = Path(args.log_dir) / "debug" / split / f"{rel}.json"
                        atomic_write(dbg, json.dumps(results, ensure_ascii=False, indent=1))
                    done += 1
                except Exception as e:  # noqa: BLE001
                    failed.append(f"{img}\t{e}")
                    print(f"[dist] FAILED {img}: {e}", flush=True)

        elapsed = time.time() - t0
        summary = stats.summary(elapsed)
        summary.update(
            {
                "stats_scope": "processed images only (skipped existing ones are not re-read)",
                "dataset_root": str(root),
                "labels_dist": str(out_dir),
                "teacher": str(weights),
                "teacher_md5": md5,
                "imgsz": args.imgsz,
                "max_dist": args.max_dist,
                "min_valid_pixels": args.min_valid_pixels,
                "written": done,
                "skipped_existing": skipped,
                "failed": len(failed),
            }
        )
        stats_dir = Path(args.stats_dir)
        stats_dir.mkdir(parents=True, exist_ok=True)
        (stats_dir / f"{split}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        write_csv(stats_dir / f"{split}_boxes.csv", stats.rows, args.max_dist)
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"dist_labels_{split}.log").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        (log_dir / f"failed_{split}.txt").write_text("\n".join(failed) + ("\n" if failed else ""))
        print(
            f"[done] {split}: images={summary['images']} boxes={summary['boxes_total']} "
            f"valid={summary['boxes_valid']} ignore={summary['boxes_ignore']} "
            f"written={done} skipped={skipped} failed={len(failed)} "
            f"elapsed={summary['elapsed_sec']}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
