#!/usr/bin/env python3
"""Mask non-own-domain distance labels to -1 (detection labels untouched).

Reads labels_dist_cal/{train,val}; own_video + own_camera files are copied
byte-for-byte; every other file keeps its first 5 columns verbatim and gets
d_norm = -1 (distance ignore). Output: labels_dist_own/. Originals untouched.

Purpose: distance supervision only where labels are camera-calibrated; coco/fpv
keep full detection supervision but no distance gradient (cross-domain scale
conflict removed).
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

TOKEN_RE = re.compile(r"\S+")
OWN_RX = re.compile(r"^(video\d+(_\d+min)?_frame_\d+|camera\d+-\d+_frame_\d+)$")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="/mnt/data2/hzh/PF_11.9")
    p.add_argument("--src-subdir", default="labels_dist_cal")
    p.add_argument("--dst-subdir", default="labels_dist_own")
    p.add_argument("--splits", default="train,val")
    p.add_argument("--stats-dir", default="")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    ds = Path(args.dataset_root)
    src_root, dst_root = ds / args.src_subdir, ds / args.dst_subdir
    stats_dir = Path(args.stats_dir) if args.stats_dir else Path.cwd() / "stats" / "detdist_own"
    if dst_root.exists() and not args.overwrite:
        sys.exit(f"refusing to overwrite existing {dst_root} (use --overwrite)")
    stats_dir.mkdir(parents=True, exist_ok=True)
    summary = {"src": str(src_root), "dst": str(dst_root), "splits": {}}

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        src_dir, dst_dir = src_root / split, dst_root / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        st = {"files": 0, "own_copied": 0, "masked_files": 0, "boxes_masked": 0,
              "already_ignore": 0, "own_boxes_kept": 0}
        errors = []
        for f in sorted(src_dir.glob("*.txt")):
            st["files"] += 1
            dst = dst_dir / f.name
            raw = f.read_bytes()
            if OWN_RX.match(f.stem):
                shutil.copyfile(f, dst)
                st["own_copied"] += 1
                st["own_boxes_kept"] += sum(
                    1 for ln in raw.decode().splitlines() if ln.strip() and float(ln.split()[5]) >= 0
                )
                continue
            text = raw.decode()
            eol = "\r\n" if b"\r\n" in raw else "\n"
            trailing = text.endswith("\n")
            out = []
            for i, line in enumerate(text.splitlines()):
                if not line.strip():
                    out.append(line)
                    continue
                spans = [m.span() for m in TOKEN_RE.finditer(line)]
                if len(spans) != 6:
                    errors.append(f"{f.name}:{i+1} cols={len(spans)}")
                    out.append(line)
                    continue
                prefix = line[: spans[4][1]]
                if float(line[spans[5][0] : spans[5][1]]) < 0:
                    st["already_ignore"] += 1
                    out.append(line)
                else:
                    st["boxes_masked"] += 1
                    out.append(prefix + " -1")
            data = eol.join(out) + (eol if trailing else "")
            tmp = dst.with_name(dst.name + f".tmp.{os.getpid()}")
            tmp.write_bytes(data.encode())
            os.replace(tmp, dst)
            st["masked_files"] += 1
        # verify: file sets, own byte-identical, masked first-5 identical + all -1
        src_files = {x.name for x in src_dir.glob("*.txt")}
        dst_files = {x.name for x in dst_dir.glob("*.txt")}
        if src_files != dst_files:
            errors.append("file set mismatch")
        for name in sorted(src_files & dst_files):
            sb = (src_dir / name).read_bytes()
            db = (dst_dir / name).read_bytes()
            if OWN_RX.match(Path(name).stem):
                if sb != db:
                    errors.append(f"own file changed: {name}")
                continue
            sl, dl = sb.decode().splitlines(), db.decode().splitlines()
            if len(sl) != len(dl):
                errors.append(f"line count: {name}")
                continue
            for i, (s, d) in enumerate(zip(sl, dl)):
                if not s.strip():
                    continue
                ss = [m.span() for m in TOKEN_RE.finditer(s)]
                dd = [m.span() for m in TOKEN_RE.finditer(d)]
                if s[: ss[4][1]] != d[: dd[4][1]]:
                    errors.append(f"prefix changed: {name}:{i+1}")
                if d[dd[5][0] : dd[5][1]] != "-1":
                    errors.append(f"not masked: {name}:{i+1}")
        st["errors"] = errors[:20]
        st["n_errors"] = len(errors)
        summary["splits"][split] = st
        print(f"[{split}] files={st['files']} own_copied={st['own_copied']} "
              f"own_boxes_kept={st['own_boxes_kept']} masked_files={st['masked_files']} "
              f"boxes_masked={st['boxes_masked']} already_ignore={st['already_ignore']} "
              f"errors={len(errors)}")
        if errors:
            print("FIRST ERRORS:", errors[:5])
    summary["all_checks_passed"] = all(s["n_errors"] == 0 for s in summary["splits"].values())
    (stats_dir / "own_mask_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary -> {stats_dir / 'own_mask_summary.json'}")
    print("ALL CHECKS PASSED" if summary["all_checks_passed"] else "CHECKS FAILED")
    sys.exit(0 if summary["all_checks_passed"] else 1)


if __name__ == "__main__":
    main()