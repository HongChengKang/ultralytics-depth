#!/usr/bin/env python3
"""Affine-calibrate own_video distance labels into a new labels directory.

d_corr = (d_norm * max_dist - b) / a ; d_norm' = d_corr / max_dist
Only files whose stem matches --pattern are transformed; every other file is
copied byte-for-byte. The first 5 columns of transformed lines are preserved
verbatim. Original labels are never modified. No training is involved.
"""
import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

TOKEN_RE = re.compile(r"\S+")
DEFAULT_PATTERN = r"^video\d+(_\d+min)?_frame_\d+$"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default="/mnt/data2/hzh/PF_11.9")
    p.add_argument("--labels-subdir", default="labels_dist")
    p.add_argument("--output-subdir", default="labels_dist_cal")
    p.add_argument("--splits", default="train,val")
    p.add_argument("--a", type=float, default=1.38, help="teacher = a*true + b")
    p.add_argument("--b", type=float, default=0.37)
    p.add_argument("--max-dist", type=float, default=15.0, help="metres, calibrated scale")
    p.add_argument("--pattern", default=DEFAULT_PATTERN)
    p.add_argument("--stats-dir", default="")
    p.add_argument("--limit", type=int, default=0, help="process at most N files per split (smoke)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="compute stats, write nothing")
    return p.parse_args()


def line_eol(raw_bytes):
    return "\r\n" if b"\r\n" in raw_bytes else "\n"


def new_counters():
    c = {k: 0 for k in ("files_total", "files_matched", "files_copied",
                        "matched_unchanged", "boxes_transformed", "ignore_kept",
                        "new_ignore_nonpositive", "new_ignore_gt_max")}
    for k in ("d_m_before", "d_m_after", "k_before", "k_after"):
        c[k] = []
    return c


def merge_counters(dst, src):
    for k, v in src.items():
        if isinstance(v, list):
            dst[k].extend(v)
        else:
            dst[k] += v


def transform_file(src, dst, rx, a, b, max_dist, img_size, rows, counters):
    raw = src.read_bytes()
    text = raw.decode("utf-8")
    eol = line_eol(raw)
    trailing = text.endswith("\n")
    out_lines = []
    changed = False
    for line in text.splitlines():
        if not line.strip():
            out_lines.append(line)
            continue
        spans = [m.span() for m in TOKEN_RE.finditer(line)]
        if len(spans) != 6:
            raise ValueError(f"{src}: expected 6 columns, got {len(spans)}")
        prefix = line[: spans[4][1]]
        tok = line[spans[5][0] : spans[5][1]]
        d_norm = float(tok)
        if d_norm < 0:
            counters["ignore_kept"] += 1
            out_lines.append(line)
            continue
        d_m = d_norm * max_dist
        d_corr = (d_m - b) / a
        if d_corr <= 0:
            counters["new_ignore_nonpositive"] += 1
            out_lines.append(prefix + " -1")
            changed = True
            continue
        if d_corr > max_dist:
            counters["new_ignore_gt_max"] += 1
            out_lines.append(prefix + " -1")
            changed = True
            continue
        new_norm = round(d_corr / max_dist, 6)
        out_lines.append(f"{prefix} {new_norm:.6f}")
        changed = True
        counters["boxes_transformed"] += 1
        counters["d_m_before"].append(d_m)
        counters["d_m_after"].append(d_corr)
        if img_size is not None:
            w, h = img_size
            scale = min(640.0 / w, 640.0 / h)
            h_canvas = float(line[spans[4][0] : spans[4][1]]) * h * scale
            if h_canvas > 0:
                counters["k_before"].append(d_m * h_canvas)
                counters["k_after"].append(d_corr * h_canvas)
        rows.append(
            [src.stem, line[: spans[0][1]], f"{d_m:.4f}", f"{d_corr:.4f}", f"{new_norm:.6f}"]
        )
    if not changed:
        shutil.copyfile(src, dst)
        counters["matched_unchanged"] += 1
        return
    data = eol.join(out_lines) + (eol if trailing else "")
    tmp = dst.with_name(dst.name + f".tmp.{os.getpid()}")
    tmp.write_bytes(data.encode("utf-8"))
    os.replace(tmp, dst)


def median(xs):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def quantile(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def main():
    args = parse_args()
    ds = Path(args.dataset_root)
    src_root = ds / args.labels_subdir
    dst_root = ds / args.output_subdir
    rx = re.compile(args.pattern)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    stats_dir = Path(args.stats_dir) if args.stats_dir else Path.cwd() / "stats" / "detdist_cal"
    if not args.dry_run:
        stats_dir.mkdir(parents=True, exist_ok=True)
        if dst_root.exists() and not args.overwrite:
            sys.exit(f"refusing to overwrite existing {dst_root} (use --overwrite)")
        for s in splits:
            (dst_root / s).mkdir(parents=True, exist_ok=True)

    from PIL import Image

    summary = {
        "calibration": {"a": args.a, "b": args.b, "max_dist_m": args.max_dist,
                        "formula": "d_corr = (d_norm*max_dist - b) / a"},
        "pattern": args.pattern,
        "src": str(src_root), "dst": str(dst_root),
        "dry_run": args.dry_run, "splits": {},
    }
    t0 = time.time()
    for split in splits:
        src_dir, dst_dir = src_root / split, dst_root / split
        files = sorted(src_dir.glob("*.txt"))
        if args.limit:
            files = files[: args.limit]
        img_dir = ds / "images" / split
        c = new_counters()
        k_by_res = {}
        rows = []
        for f in files:
            c["files_total"] += 1
            dst = dst_dir / f.name
            if not rx.match(f.stem):
                c["files_copied"] += 1
                if not args.dry_run:
                    shutil.copyfile(f, dst)
                continue
            c["files_matched"] += 1
            img_size = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                im = img_dir / (f.stem + ext)
                if im.exists():
                    with Image.open(im) as pil:
                        img_size = pil.size
                    break
            if img_size is None:
                print(f"WARN: no image found for {f.stem}, K stats skipped")
            fc = new_counters()
            if args.dry_run:
                transform_file_inmem(f, rx, args.a, args.b, args.max_dist, img_size, rows, fc)
            else:
                transform_file(f, dst, rx, args.a, args.b, args.max_dist, img_size, rows, fc)
            merge_counters(c, fc)
            if img_size is not None and fc["k_after"]:
                key = f"{img_size[0]}x{img_size[1]}"
                k_by_res.setdefault(key, []).extend(fc["k_after"])
        st = {k: v for k, v in c.items() if not isinstance(v, list)}
        for k in ("d_m_before", "d_m_after", "k_before", "k_after"):
            xs = c[k]
            st[k] = {"n": len(xs), "median": median(xs),
                     "p25": quantile(xs, 0.25), "p75": quantile(xs, 0.75),
                     "min": min(xs) if xs else None, "max": max(xs) if xs else None}
        st["k_after_by_resolution"] = {
            res: {"n": len(v), "median": median(v)} for res, v in sorted(k_by_res.items())
        }
        summary["splits"][split] = st
        if not args.dry_run:
            with open(stats_dir / f"{split}_cal_boxes.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["stem", "cls", "d_m_before", "d_m_after", "d_norm_after"])
                w.writerows(rows)
        print(f"[{split}] files={c['files_total']} matched={c['files_matched']} "
              f"copied={c['files_copied']} boxes_transformed={c['boxes_transformed']} "
              f"ignore_kept={c['ignore_kept']} new_ignore={c['new_ignore_nonpositive'] + c['new_ignore_gt_max']}")
        print(f"[{split}] d_m median {st['d_m_before']['median']} -> {st['d_m_after']['median']} | "
              f"K median {st['k_before']['median']} -> {st['k_after']['median']}")
        for res, v in st["k_after_by_resolution"].items():
            print(f"[{split}] K after @{res}: median={v['median']:.1f} n={v['n']}")
    summary["elapsed_s"] = round(time.time() - t0, 1)
    if not args.dry_run:
        with open(stats_dir / "cal_summary.json", "w") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        print(f"summary -> {stats_dir / 'cal_summary.json'}")
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


def transform_file_inmem(src, rx, a, b, max_dist, img_size, rows, counters):
    text = src.read_text()
    for line in text.splitlines():
        if not line.strip():
            continue
        spans = [m.span() for m in TOKEN_RE.finditer(line)]
        if len(spans) != 6:
            raise ValueError(f"{src}: expected 6 columns, got {len(spans)}")
        tok = line[spans[5][0] : spans[5][1]]
        d_norm = float(tok)
        if d_norm < 0:
            counters["ignore_kept"] += 1
            continue
        d_m = d_norm * max_dist
        d_corr = (d_m - b) / a
        if d_corr <= 0:
            counters["new_ignore_nonpositive"] += 1
            continue
        if d_corr > max_dist:
            counters["new_ignore_gt_max"] += 1
            continue
        counters["boxes_transformed"] += 1
        counters["d_m_before"].append(d_m)
        counters["d_m_after"].append(d_corr)
        if img_size is not None:
            w, h = img_size
            scale = min(640.0 / w, 640.0 / h)
            h_canvas = float(line[spans[4][0] : spans[4][1]]) * h * scale
            if h_canvas > 0:
                counters["k_before"].append(d_m * h_canvas)
                counters["k_after"].append(d_corr * h_canvas)
        rows.append([src.stem, line[: spans[0][1]], f"{d_m:.4f}", f"{d_corr:.4f}",
                     f"{round(d_corr / max_dist, 6):.6f}"])


if __name__ == "__main__":
    main()