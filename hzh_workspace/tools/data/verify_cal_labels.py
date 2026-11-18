#!/usr/bin/env python3
"""Independently verify labels_dist_cal against labels_dist.

Checks: file counts, non-own_video files byte-identical, own_video first-5
columns byte-identical, line counts, d_norm in [-1,1], affine relation holds
within rounding tolerance, ignore accounting.
"""
import argparse
import json
import re
import sys
from pathlib import Path

TOKEN_RE = re.compile(r"\S+")
DEFAULT_PATTERN = r"^video\d+(_\d+min)?_frame_\d+$"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="/mnt/data2/hzh/PF_11.9")
    p.add_argument("--src-subdir", default="labels_dist")
    p.add_argument("--dst-subdir", default="labels_dist_cal")
    p.add_argument("--splits", default="train,val")
    p.add_argument("--a", type=float, default=1.38)
    p.add_argument("--b", type=float, default=0.37)
    p.add_argument("--max-dist", type=float, default=15.0)
    p.add_argument("--pattern", default=DEFAULT_PATTERN)
    p.add_argument("--report", default="")
    args = p.parse_args()
    ds = Path(args.dataset_root)
    rx = re.compile(args.pattern)
    report = {"all_checks_passed": True, "splits": {}}

    def fail(split, msg):
        report["all_checks_passed"] = False
        report["splits"][split].setdefault("errors", []).append(msg)
        print(f"FAIL [{split}] {msg}")

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        src_dir = ds / args.src_subdir / split
        dst_dir = ds / args.dst_subdir / split
        st = {"errors": []}
        report["splits"][split] = st
        src_files = {f.name for f in src_dir.glob("*.txt")}
        dst_files = {f.name for f in dst_dir.glob("*.txt")}
        st["src_files"], st["dst_files"] = len(src_files), len(dst_files)
        if src_files != dst_files:
            fail(split, f"file set mismatch: missing={len(src_files - dst_files)} orphan={len(dst_files - src_files)}")
        n_copied = n_matched = n_lines = n_bad_prefix = n_bad_range = n_affine = 0
        n_ignore_src = n_ignore_dst = n_new_ignore = 0
        max_dev = 0.0
        for name in sorted(src_files & dst_files):
            sf, df = src_dir / name, dst_dir / name
            sb, db = sf.read_bytes(), df.read_bytes()
            if not rx.match(Path(name).stem):
                if sb != db:
                    fail(split, f"non-target file changed: {name}")
                n_copied += 1
                continue
            n_matched += 1
            sl, dl = sb.decode().splitlines(), db.decode().splitlines()
            if len(sl) != len(dl):
                fail(split, f"line count mismatch: {name}")
                continue
            for i, (s, d) in enumerate(zip(sl, dl)):
                if not s.strip():
                    if s != d:
                        fail(split, f"blank line changed: {name}:{i+1}")
                    continue
                n_lines += 1
                ss = [m.span() for m in TOKEN_RE.finditer(s)]
                dd = [m.span() for m in TOKEN_RE.finditer(d)]
                if len(ss) != 6 or len(dd) != 6:
                    fail(split, f"column count != 6: {name}:{i+1}")
                    continue
                if s[: ss[4][1]] != d[: dd[4][1]]:
                    n_bad_prefix += 1
                    fail(split, f"first-5 columns changed: {name}:{i+1}")
                sv = float(s[ss[5][0]:ss[5][1]])
                dv = float(d[dd[5][0]:dd[5][1]])
                if not (-1.0 <= dv <= 1.0):
                    n_bad_range += 1
                    fail(split, f"d_norm out of range: {name}:{i+1} -> {dv}")
                if sv < 0:
                    n_ignore_src += 1
                    if dv != -1:
                        fail(split, f"ignore not preserved: {name}:{i+1}")
                    continue
                expect = (sv * args.max_dist - args.b) / args.a
                if expect <= 0 or expect > args.max_dist:
                    n_new_ignore += 1
                    if dv != -1:
                        fail(split, f"expected new ignore: {name}:{i+1}")
                    continue
                dev = abs(dv - expect / args.max_dist)
                max_dev = max(max_dev, dev)
                if dev > 1.5e-6:
                    n_affine += 1
                    fail(split, f"affine mismatch: {name}:{i+1} dev={dev:.2e}")
                if dv == -1:
                    n_ignore_dst += 1
        st.update({"files_byte_identical": n_copied, "files_transformed": n_matched,
                   "lines_checked": n_lines, "bad_prefix": n_bad_prefix,
                   "bad_range": n_bad_range, "affine_mismatch": n_affine,
                   "ignore_src": n_ignore_src, "new_ignore": n_new_ignore,
                   "max_affine_dev": max_dev})
        print(f"[{split}] files={len(src_files)} copied_identical={n_copied} transformed={n_matched} "
              f"lines={n_lines} ignore_src={n_ignore_src} new_ignore={n_new_ignore} "
              f"max_affine_dev={max_dev:.2e}")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"report -> {args.report}")
    print("ALL CHECKS PASSED" if report["all_checks_passed"] else "CHECKS FAILED")
    sys.exit(0 if report["all_checks_passed"] else 1)


if __name__ == "__main__":
    main()