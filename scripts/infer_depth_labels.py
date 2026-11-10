#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Infer per-pixel metric depth with yolo26x-depth.pt and store .npy labels.

What it does
------------
For every image under ``<source>/images/<split>/`` it runs the depth teacher and
writes one ``.npy`` per image, mirroring the image tree:

    <source>/images/val/a.jpg  ->  <output-dir>/val/a.npy

Array contract (see manifest.json for the machine-readable version):

* dtype  : float32
* shape  : exactly (H, W) of the *original* image (the predictor already
           bilinearly restores the model-resolution map to the source size)
* unit   : metre -- the Depth head predicts ``exp(clamp(logit))`` and the
           weight file carries its log-affine calibration (``cal_a``/``cal_b``);
           it is the teacher's own metric scale, NOT a per-camera calibration
           (this dataset has no intrinsics / scale factor)
* norm   : none (raw metres are stored, no clipping, no per-image normalisation)

Safety / idempotency
--------------------
* never writes into ``images/`` or ``labels/``; only ``<output-dir>``
* existing ``.npy`` files are skipped by default (``--overwrite`` to recompute)
* atomic writes (``.tmp`` + ``os.replace``) so an interrupted run leaves no
  half-written array
* a single bad image never aborts the run: it is recorded in ``failures`` and
  in the manifest, and the process keeps going

Examples
--------
    python scripts/infer_depth_labels.py \
        --weights weights/yolo26x-depth.pt \
        --source /mnt/data2/hzh/PF_11.9 \
        --output-dir /mnt/data2/hzh/PF_11.9/depth_labels \
        --device auto --batch 1 --save-vis

    # re-run only the checks against what is already on disk
    python scripts/infer_depth_labels.py --source /mnt/data2/hzh/PF_11.9 \
        --output-dir /mnt/data2/hzh/PF_11.9/depth_labels --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(str(tmp), str(path))


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt(v: Optional[float], nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


# --------------------------------------------------------------------------- #
# dataset discovery
# --------------------------------------------------------------------------- #
def discover_dataset(source: Path) -> Tuple[Path, Dict[str, List[Path]]]:
    """Return (images_root, {split: [image paths]}).

    Layouts understood, in order:
      1. <source>/images/<split>/**      (this dataset)
      2. <source>/<split>/**             (images directly under split dirs)
      3. <source>/**                     (flat folder -> split "")
    """
    candidates = [source / "images", source / "Images", source / "images_all"]
    images_root = next((p for p in candidates if p.is_dir()), None)

    if images_root is not None:
        splits: Dict[str, List[Path]] = {}
        for d in sorted(images_root.iterdir()):
            if not d.is_dir():
                continue
            files = sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            if files:
                splits[d.name] = files
        if splits:
            return images_root, splits

    # fallback 2: split dirs directly under source (train/ val/ test/)
    splits = {}
    for d in sorted(source.iterdir()):
        if not d.is_dir() or d.name in {"labels", "depth_labels"}:
            continue
        files = sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if files:
            splits[d.name] = files
    if splits:
        return source, splits

    # fallback 3: flat image folder
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise SystemExit(f"no images found under {source}")
    return source, {"": files}


def find_labels_root(source: Path, images_root: Path) -> Optional[Path]:
    for cand in (source / "labels", images_root.parent / "labels"):
        if cand.is_dir():
            return cand
    return None


def data_yaml_of(source: Path) -> Optional[Path]:
    for name in ("data.yaml", "data.yml", "dataset.yaml"):
        p = source / name
        if p.is_file():
            return p
    return None


def calibration_files(source: Path, images_root: Path) -> List[str]:
    """Best-effort search for intrinsics / scale files (never image payloads)."""
    hits: List[str] = []
    for p in source.rglob("*"):
        if not p.is_file() or p.suffix.lower() in IMAGE_EXTS:
            continue
        try:
            p.relative_to(images_root)
            continue  # an image tree holds no calibration sidecars we care about
        except ValueError:
            pass
        low = p.name.lower()
        if any(k in low for k in ("calib", "intrins", "homograph", "scale_factor")):
            hits.append(str(p.relative_to(source)))
    return sorted(hits)[:10]


# --------------------------------------------------------------------------- #
# depth extraction
# --------------------------------------------------------------------------- #
def depth_to_numpy(result: Any) -> Tuple[np.ndarray, bool]:
    """Return (float32 (H,W) depth, resized_flag)."""
    depth = getattr(result, "depth", None)
    if depth is None:
        raise RuntimeError("model returned no depth map (result.depth is None)")
    data = depth.data
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 3:  # (1,H,W) -> (H,W)
        arr = arr[0]
    if arr.ndim != 2:
        raise RuntimeError(f"unexpected depth shape {arr.shape}")

    h, w = result.orig_shape[:2]
    resized = False
    if arr.shape != (h, w):
        import cv2

        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
        resized = True
    return np.ascontiguousarray(arr, dtype=np.float32), resized


def array_stats(arr: np.ndarray) -> Dict[str, Optional[float]]:
    finite = np.isfinite(arr)
    n = int(arr.size)
    if not finite.any():
        return {"min": None, "median": None, "max": None, "valid_ratio": 0.0}
    vals = arr[finite]
    pos = vals[vals > 0]
    return {
        "min": float(pos.min()) if pos.size else float(vals.min()),
        "median": float(np.median(pos)) if pos.size else float(np.median(vals)),
        "max": float(pos.max()) if pos.size else float(vals.max()),
        "valid_ratio": float(pos.size) / float(n) if n else 0.0,
    }


def save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)
    os.replace(str(tmp), str(path))


def load_npy_stats(path: Path) -> Tuple[Tuple[int, int], Dict[str, Optional[float]], str]:
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    stats = array_stats(np.asarray(arr))
    return tuple(arr.shape), stats, str(arr.dtype)


# --------------------------------------------------------------------------- #
# visualisation (optional, only under <output-dir>/vis)
# --------------------------------------------------------------------------- #
def save_vis(img_path: Path, depth: np.ndarray, out_path: Path, stats: Dict[str, Any]) -> None:
    import cv2

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cv2 could not read {img_path}")
    if bgr.shape[:2] != depth.shape[:2]:
        depth = cv2.resize(depth, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    finite = np.isfinite(depth) & (depth > 0)
    lo, hi = np.percentile(depth[finite], [2, 98]) if finite.any() else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1e-3
    norm = np.clip((depth - lo) / (hi - lo), 0, 1)
    norm[~finite] = 0
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    blend = cv2.addWeighted(bgr, 0.45, heat, 0.55, 0)
    text = f"min {fmt(stats.get('min'),2)}  med {fmt(stats.get('median'),2)}  max {fmt(stats.get('max'),2)} m"
    cv2.rectangle(blend, (0, 0), (min(blend.shape[1], 430), 26), (0, 0, 0), -1)
    cv2.putText(blend, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp.jpg")
    cv2.imwrite(str(tmp), blend, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    os.replace(str(tmp), str(out_path))


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def load_manifest(path: Path) -> Dict[str, Any]:
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:  # corrupted manifest must not block a run
            log(f"WARNING: could not parse {path} ({exc}); starting a fresh manifest")
    return {"schema_version": SCHEMA_VERSION, "images": {}, "runs": []}


def build_model_block(model: Any, weights: Path, device_label: str, imgsz_used: int) -> Dict[str, Any]:
    import ultralytics
    import torch

    head = None
    try:
        head = model.model.model[-1]
    except Exception:
        pass
    cal_a = float(head.cal_a) if head is not None and hasattr(head, "cal_a") else None
    cal_b = float(head.cal_b) if head is not None and hasattr(head, "cal_b") else None
    return {
        "path": str(weights),
        "md5": md5sum(weights),
        "bytes": weights.stat().st_size,
        "task": getattr(model, "task", None),
        "names": {int(k): str(v) for k, v in getattr(model, "names", {}).items()},
        "imgsz_requested": imgsz_used,
        "head": type(head).__name__ if head is not None else None,
        "cal_a": cal_a,
        "cal_b": cal_b,
        "calibration_note": (
            "log-affine scale calibration baked into the weight file: depth = exp(cal_a * logit + cal_b). "
            "It is the teacher's own metric scale, not a per-camera calibration."
        ),
        "device": device_label,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
    }


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def run_inference(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    t_start = time.time()
    source = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"--source is not a directory: {source}")
    if not weights.is_file():
        raise SystemExit(f"--weights not found: {weights}")

    images_root, splits = discover_dataset(source)
    if args.splits:
        keep = {s.strip() for s in args.splits.split(",") if s.strip()}
        splits = {k: v for k, v in splits.items() if k in keep}
        if not splits:
            raise SystemExit(f"none of the requested splits {sorted(keep)} exist under {images_root}")

    if args.limit and args.limit > 0:
        splits = {k: v[: args.limit] for k, v in splits.items()}

    total_imgs = sum(len(v) for v in splits.values())
    log(f"source      : {source}")
    log(f"images root : {images_root}")
    split_brief = ", ".join(f"{k or '<flat>'}={len(v)}" for k, v in splits.items())
    log(f"splits      : {split_brief}  (total {total_imgs})")
    log(f"output      : {output_dir}")
    log(f"weights     : {weights}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / args.manifest_name
    manifest = load_manifest(manifest_path)
    manifest.setdefault("images", {})
    manifest.setdefault("runs", [])

    # ---------- model ----------
    model = YOLO(str(weights))
    if getattr(model, "task", None) != "depth":
        raise SystemExit(f"expected a depth model, got task={getattr(model, 'task', None)}")

    import torch

    if args.device == "auto":
        device_arg = 0 if torch.cuda.is_available() else "cpu"
    elif args.device.isdigit():
        device_arg = int(args.device)
    else:
        device_arg = args.device
    device_label = f"cuda:{device_arg}" if isinstance(device_arg, int) else str(device_arg)
    gpu_name = torch.cuda.get_device_name(device_arg) if isinstance(device_arg, int) else None

    model_imgsz = int(model.model.args.get("imgsz", 768) or 768)
    imgsz_used = int(args.imgsz) if args.imgsz else model_imgsz
    if args.imgsz and int(args.imgsz) != model_imgsz:
        log(f"WARNING: --imgsz {args.imgsz} differs from the model default {model_imgsz}")

    model_block = build_model_block(model, weights, device_label, imgsz_used)
    log(f"model       : task={model_block['task']} names={model_block['names']} imgsz={imgsz_used} "
        f"head={model_block['head']} cal_a={model_block['cal_a']} cal_b={model_block['cal_b']}")
    log(f"device      : {device_label} {gpu_name or ''}")

    labels_root = find_labels_root(source, images_root)
    yaml_path = data_yaml_of(source)
    calib = calibration_files(source, images_root)
    log(f"labels root : {labels_root}   data yaml: {yaml_path}   calibration files: {calib or 'none'}")

    run_record: Dict[str, Any] = {
        "started_at": now_iso(),
        "elapsed_sec": None,
        "device": device_label,
        "gpu": gpu_name,
        "batch": int(args.batch),
        "imgsz": imgsz_used,
        "skip_existing": bool(args.skip_existing and not args.overwrite),
        "save_vis": bool(args.save_vis),
        "written": 0,
        "skipped": 0,
        "failed": 0,
        "images": total_imgs,
    }
    failures: List[Dict[str, str]] = []
    written = skipped = 0
    vis_done: Dict[str, int] = {k: 0 for k in splits}

    # ultralytics writes its run dir relative to cwd; park that outside the dataset
    run_root = Path(tempfile.mkdtemp(prefix="depth_predict_runs_", dir=tempfile.gettempdir()))

    for split, files in splits.items():
        log(f"--- split {split or '<flat>'}: {len(files)} images")
        t_split = time.time()
        batch_paths: List[Path] = []
        batch_meta: List[Dict[str, Any]] = []

        def flush(batch_paths: List[Path], batch_meta: List[Dict[str, Any]]) -> None:
            nonlocal written, skipped
            if not batch_paths:
                return

            pairs: List[Tuple[Dict[str, Any], Any]] = []
            try:
                results = list(
                    model.predict(
                        [str(p) for p in batch_paths],
                        device=device_arg,
                        imgsz=imgsz_used,
                        batch=int(args.batch),
                        save=False,
                        verbose=False,
                        project=str(run_root),
                        name="predict",
                        exist_ok=True,
                        stream=False,
                    )
                )
                if len(results) != len(batch_meta):
                    raise RuntimeError(f"got {len(results)} results for {len(batch_meta)} images")
                pairs = list(zip(batch_meta, results))
            except Exception as exc:
                log(f"batch of {len(batch_paths)} failed ({exc}) -> retrying one by one")
                pairs = []
                for meta in batch_meta:
                    try:
                        one = model.predict(
                            str(meta["path"]), device=device_arg, imgsz=imgsz_used, save=False,
                            verbose=False, project=str(run_root), name="predict", exist_ok=True,
                        )
                        pairs.append((meta, one[0]))
                    except Exception as exc2:
                        failures.append({"image": meta["rel"], "error": f"{type(exc2).__name__}: {exc2}"})
                        run_record["failed"] += 1
                        log(f"FAILED {meta['rel']}: {exc2}")

            for meta, res in pairs:
                rel_img = meta["rel"]
                npy_rel = meta["npy_rel"]
                try:
                    arr, resized = depth_to_numpy(res)
                    save_npy(output_dir / npy_rel, arr)
                    stats = array_stats(arr)
                    manifest["images"][rel_img] = {
                        "npy": npy_rel,
                        "orig_size": [int(res.orig_shape[0]), int(res.orig_shape[1])],
                        "shape": [int(arr.shape[0]), int(arr.shape[1])],
                        "dtype": "float32",
                        "min": stats["min"],
                        "median": stats["median"],
                        "max": stats["max"],
                        "valid_ratio": stats["valid_ratio"],
                        "resized": resized,
                        "generated_at": now_iso(),
                    }
                    written += 1
                    run_record["written"] += 1

                    if args.save_vis and vis_done[split] < int(args.vis_count):
                        try:
                            vis_path = output_dir / "vis" / f"{split or 'flat'}_{Path(rel_img).stem}.jpg"
                            save_vis(Path(res.path), arr, vis_path, stats)
                            vis_done[split] += 1
                        except Exception as exc:
                            log(f"vis failed for {rel_img}: {exc}")
                except Exception as exc:
                    failures.append({"image": rel_img, "error": f"{type(exc).__name__}: {exc}"})
                    run_record["failed"] += 1
                    log(f"FAILED {rel_img}: {exc}")

        for i, img in enumerate(files, 1):
            rel_img = img.relative_to(images_root).as_posix()
            npy_rel = str(Path(rel_img).with_suffix(".npy"))
            dst = output_dir / npy_rel

            if dst.is_file() and not args.overwrite:
                entry = manifest["images"].get(rel_img)
                if entry is None:
                    try:
                        shape, stats, dtype = load_npy_stats(dst)
                        entry = {
                            "npy": npy_rel,
                            "orig_size": [int(shape[0]), int(shape[1])],
                            "shape": [int(shape[0]), int(shape[1])],
                            "dtype": dtype,
                            "min": stats["min"],
                            "median": stats["median"],
                            "max": stats["max"],
                            "valid_ratio": stats["valid_ratio"],
                            "resized": None,
                            "generated_at": "recovered-from-disk",
                        }
                    except Exception as exc:
                        log(f"existing npy unreadable ({dst}): {exc} -> recomputing")
                        entry = None
                if entry is not None:
                    manifest["images"][rel_img] = entry
                    skipped += 1
                    run_record["skipped"] += 1
                    continue

            batch_paths.append(img)
            batch_meta.append({"rel": rel_img, "npy_rel": npy_rel, "path": img})
            if len(batch_paths) >= max(1, int(args.batch)):
                flush(batch_paths, batch_meta)
                batch_paths, batch_meta = [], []

            if i % 500 == 0:
                log(f"  {i}/{len(files)}  written={written} skipped={skipped} failed={run_record['failed']}")

        flush(batch_paths, batch_meta)
        log(f"--- split {split or '<flat>'} done in {time.time() - t_split:.1f}s "
            f"(written={written} skipped={skipped} failed={run_record['failed']})")

    elapsed = time.time() - t_start
    run_record.update({"finished_at": now_iso(), "elapsed_sec": round(elapsed, 2)})

    # numeric overview over everything we know about (new + carried over)
    entries = manifest["images"]
    medians = [e["median"] for e in entries.values() if e.get("median")]
    run_record["images_in_manifest"] = len(entries)

    manifest.update({
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "model": model_block,
        "dataset": {
            "root": str(source),
            "images_root": str(images_root),
            "splits": {k or "flat": len(v) for k, v in splits.items()},
            "labels_root": str(labels_root) if labels_root else None,
            "data_yaml": str(yaml_path) if yaml_path else None,
            "calibration_files": calib,
            "has_calibration": bool(calib),
        },
        "output": {
            "root": str(output_dir),
            "layout": "<split>/<image stem>.npy",
            "dtype": "float32",
            "shape": "(H, W) of the original image",
            "unit": "meter",
            "normalization": "none",
            "unit_note": (
                "Metres as produced by the teacher's Depth head with its baked log-affine calibration. "
                "No camera intrinsics / scale factor exist in this dataset, so this is model-metric, not "
                "survey-calibrated, distance."
            ),
        },
        "params": {
            "weights": str(weights),
            "source": str(source),
            "output_dir": str(output_dir),
            "batch": int(args.batch),
            "device": device_label,
            "imgsz": imgsz_used,
            "skip_existing": bool(args.skip_existing and not args.overwrite),
            "overwrite": bool(args.overwrite),
            "save_vis": bool(args.save_vis),
            "vis_count": int(args.vis_count),
            "limit": int(args.limit),
            "manifest_name": args.manifest_name,
        },
        "run": run_record,
        "summary": {
            "images_written": run_record["written"],
            "images_skipped": run_record["skipped"],
            "images_failed": run_record["failed"],
            "elapsed_sec": round(elapsed, 2),
            "img_per_sec": round((run_record["written"] + run_record["skipped"]) / elapsed, 2) if elapsed > 0 else None,
            "median_of_medians_m": float(np.median(medians)) if medians else None,
            "min_of_min_m": min((e["min"] for e in entries.values() if e.get("min")), default=None),
            "max_of_max_m": max((e["max"] for e in entries.values() if e.get("max")), default=None),
        },
        "failures": failures,
    })
    manifest["runs"] = (manifest.get("runs") or [])[-9:] + [run_record]

    json_write(manifest_path, manifest)

    log("")
    log(f"done in {elapsed:.1f}s: written={run_record['written']} skipped={run_record['skipped']} "
        f"failed={run_record['failed']}")
    if medians:
        log(f"depth overview: median-of-medians={np.median(medians):.3f} m  "
            f"min={min(e['min'] for e in entries.values() if e.get('min')):.3f}  "
            f"max={max(e['max'] for e in entries.values() if e.get('max')):.3f} m")
        mm = float(np.median(medians))
        if mm < 0.1 or mm > 200.0:
            log("WARNING: median depth outside [0.1, 200] m -- check whether the model really outputs metres")
    if failures:
        log(f"WARNING: {len(failures)} failed images; see {manifest_path} -> failures")
        for f in failures[:10]:
            log(f"  - {f['image']}: {f['error']}")
    log(f"manifest: {manifest_path}")
    return 0 if run_record["failed"] == 0 else 1


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def run_verify(args: argparse.Namespace) -> int:
    import cv2

    source = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    images_root, splits = discover_dataset(source)
    if args.splits:
        keep = {s.strip() for s in args.splits.split(",") if s.strip()}
        splits = {k: v for k, v in splits.items() if k in keep}

    manifest_path = output_dir / args.manifest_name
    manifest = load_manifest(manifest_path)

    report: Dict[str, Any] = {
        "generated_at": now_iso(),
        "source": str(source),
        "output_dir": str(output_dir),
        "splits": {},
        "totals": {"images": 0, "npy_found": 0, "missing": 0, "shape_mismatch": 0, "bad_dtype": 0},
        "samples": [],
        "warnings": [],
    }
    rng = np.random.default_rng(0)

    for split, files in splits.items():
        npy_dir = output_dir / (split or "")
        present = missing = wrong_shape = wrong_dtype = 0
        medians: List[float] = []
        mins: List[float] = []
        maxs: List[float] = []
        for img in files:
            rel = img.relative_to(images_root).with_suffix(".npy")
            p = output_dir / rel
            if not p.is_file():
                missing += 1
                continue
            try:
                arr = np.load(p, mmap_mode="r", allow_pickle=False)
            except Exception as exc:
                missing += 1
                report["warnings"].append(f"unreadable npy {rel}: {exc}")
                continue
            present += 1
            h, w = cv2.imread(str(img), cv2.IMREAD_UNCHANGED).shape[:2]
            if tuple(arr.shape) != (h, w):
                wrong_shape += 1
                report["warnings"].append(f"shape mismatch {rel}: npy={tuple(arr.shape)} image={(h, w)}")
            if np.dtype(arr.dtype) != np.dtype("float32"):
                wrong_dtype += 1
            if (present % max(1, len(files) // 40)) == 0 or present <= 3:
                st = array_stats(np.asarray(arr))
                if st["median"]:
                    medians.append(st["median"])
                if st["min"]:
                    mins.append(st["min"])
                if st["max"]:
                    maxs.append(st["max"])

        report["splits"][split or "flat"] = {
            "images": len(files),
            "npy_found": present,
            "missing": missing,
            "shape_mismatch": wrong_shape,
            "bad_dtype": wrong_dtype,
            "sampled_stats": {
                "n": len(medians),
                "median_min": min(mins, default=None),
                "median_of_medians": float(np.median(medians)) if medians else None,
                "median_max": max(maxs, default=None),
            },
        }
        for key in ("images", "npy_found", "missing", "shape_mismatch", "bad_dtype"):
            report["totals"][key] = report["totals"].get(key, 0) + report["splits"][split or "flat"][key]

    # a few deterministic sample records for eyeballing
    for split, files in splits.items():
        if not files:
            continue
        for img in rng.choice(files, size=min(3, len(files)), replace=False):
            rel = Path(img).relative_to(images_root).with_suffix(".npy")
            p = output_dir / rel
            if p.is_file():
                arr = np.load(p, mmap_mode="r", allow_pickle=False)
                st = array_stats(np.asarray(arr))
                report["samples"].append({
                    "image": str(Path(img).relative_to(images_root)), "npy": str(rel),
                    "shape": list(arr.shape), "dtype": str(arr.dtype),
                    "min": st["min"], "median": st["median"], "max": st["max"],
                    "valid_ratio": st["valid_ratio"],
                })

    out = output_dir / "verify_report.json"
    json_write(out, report)

    t = report["totals"]
    log(f"images={t['images']} npy={t['npy_found']} missing={t['missing']} "
        f"shape_mismatch={t['shape_mismatch']} bad_dtype={t['bad_dtype']}")
    for split, s in report["splits"].items():
        log(f"  {split}: images={s['images']} npy={s['npy_found']} missing={s['missing']} "
            f"shape_mismatch={s['shape_mismatch']} median~{fmt(s['sampled_stats']['median_of_medians'])} m")
    for w in report["warnings"][:10]:
        log(f"WARNING {w}")
    log(f"verify report: {out}")
    ok = t["missing"] == 0 and t["shape_mismatch"] == 0 and t["bad_dtype"] == 0
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(Path(__file__).resolve().parents[1] / "weights" / "yolo26x-depth.pt"),
                   help="depth .pt weights (default: <repo>/weights/yolo26x-depth.pt)")
    p.add_argument("--source", default="/mnt/data2/hzh/PF_11.9", help="dataset root (auto-discovers splits)")
    p.add_argument("--output-dir", default=None, help="default: <source>/depth_labels")
    p.add_argument("--splits", default="", help="comma separated subset, e.g. train,val (default: all found)")
    p.add_argument("--device", default="auto", help="auto | cpu | 0 | cuda:0")
    p.add_argument("--batch", type=int, default=1, help="inference batch size (default 1)")
    p.add_argument("--imgsz", type=int, default=0, help="override model imgsz (0 = use the model's own, 768 here)")
    p.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True,
                   help="skip images whose .npy already exists (default)")
    p.add_argument("--overwrite", dest="overwrite", action="store_true", default=False,
                   help="recompute and overwrite existing .npy")
    p.add_argument("--save-vis", dest="save_vis", action="store_true", default=False,
                   help="write a few side-by-side depth overlays into <output-dir>/vis")
    p.add_argument("--vis-count", type=int, default=8, help="max visualisations per split")
    p.add_argument("--limit", type=int, default=0, help="only the first N images per split (0 = all)")
    p.add_argument("--manifest-name", default="manifest.json", help="manifest file name inside --output-dir")
    p.add_argument("--verify-only", action="store_true", help="only check existing outputs and write verify_report.json")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        args.output_dir = str(Path(args.source).expanduser().resolve() / "depth_labels")
    if args.batch < 1:
        args.batch = 1
    if args.overwrite:
        args.skip_existing = False

    try:
        if args.verify_only:
            return run_verify(args)
        return run_inference(args)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
