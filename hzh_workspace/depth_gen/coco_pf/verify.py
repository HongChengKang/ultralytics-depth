import os, glob, sys, json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm

ROOT = "/mnt/data2/COCO_PF"
OUTDIR = sys.argv[1]
os.makedirs(OUTDIR, exist_ok=True)

def load_np(p):
    return np.array(Image.open(p))

# ---------- 1) 统计：新生成的（camera2/camera4）vs 已有的（camera1/camera3/output_fpv） ----------
def stats(pat):
    ps = sorted(glob.glob(pat))
    vals = []
    for p in ps:
        a = load_np(p)
        vals.append((int(a.min()), int(a.max()), float((a > 0).mean())))
    v = np.array(vals)
    return dict(n=len(ps),
                min_median=float(np.median(v[:, 0])), min_p1=float(np.percentile(v[:, 0], 1)),
                max_median=float(np.median(v[:, 1])), max_p99=float(np.percentile(v[:, 1], 99)),
                valid_ratio=float(v[:, 2].mean()))

new_tr = stats(f"{ROOT}/depth/train/camera[24]-1_*.png")
old_tr = stats(f"{ROOT}/depth/train/camera1-1_*.png")
new_va = stats(f"{ROOT}/depth/val/camera[24]-1_*.png")
print("NEW train:", json.dumps(new_tr, ensure_ascii=False))
print("OLD train:", json.dumps(old_tr, ensure_ascii=False))
print("NEW val  :", json.dumps(new_va, ensure_ascii=False))

# ---------- 2) 一致性：把已有 depth 的图重新跑一遍，逐像素比对 ----------
recheck = sorted(glob.glob(f"{ROOT}/depth/train/camera1-1_*.png"))[:5]
diffs = []
for p in recheck:
    stem = os.path.splitext(os.path.basename(p))[0]
    r = f"/tmp/depth_recheck/depth/train/{stem}.png"
    if not os.path.exists(r):
        continue
    a = load_np(p).astype(np.float64); b = load_np(r).astype(np.float64)
    if a.shape != b.shape:
        print("shape mismatch", stem, a.shape, b.shape); continue
    d = np.abs(a - b)
    rel = d / np.maximum(a, 1)
    diffs.append((stem, float(d.max()), float(d.mean()), float(np.median(rel)), float((d > 0).mean())))
print("\nRECHEnK (同图重跑 vs 已有):")
for row in diffs:
    print("  %-32s max|d|=%4d mean|d|=%5.2f median_rel=%.4f changed=%.3f" % row)

# ---------- 3) 对照图 ----------
picks = sorted(glob.glob(f"{ROOT}/depth/train/camera2-1_*.png"))[:3] + \
        sorted(glob.glob(f"{ROOT}/depth/train/camera4-1_*.png"))[:3] + \
        sorted(glob.glob(f"{ROOT}/depth/val/camera2-1_*.png"))[:2] + \
        sorted(glob.glob(f"{ROOT}/depth/val/camera4-1_*.png"))[:2]
TW = 640
rows = []
for p in picks:
    stem = os.path.splitext(os.path.basename(p))[0]
    split = "val" if "/val/" in p else "train"
    rgbs = glob.glob(f"{ROOT}/images/{split}/{stem}.*")
    if not rgbs:
        continue
    rgb = Image.open(rgbs[0]).convert("RGB")
    d = load_np(p).astype(np.float32)
    dn = np.clip((d - d.min()) / max(d.max() - d.min(), 1), 0, 1)
    col = (cm.turbo(dn)[..., :3] * 255).astype(np.uint8)
    dimg = Image.fromarray(col)
    h = int(rgb.size[1] * TW / rgb.size[0])
    rows.append((rgb.resize((TW, h), Image.BILINEAR), dimg.resize((TW, h), Image.BILINEAR), "rgb " + stem, "depth " + stem))

if rows:
    hh = rows[0][0].size[1]
    cols = 2
    per_row = 2  # 每行 2 组 (rgb|depth)
    nrow = (len(rows) + per_row - 1) // per_row
    W = cols * per_row * TW
    H = nrow * (hh + 18)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    for i, (a, b, ta, tb) in enumerate(rows):
        r, c = divmod(i, per_row)
        x = c * 2 * TW
        y = r * (hh + 18) + 18
        canvas.paste(a, (x, y)); canvas.paste(b, (x + TW, y))
        dr.text((x + 4, y - 14), ta, fill=(0, 0, 0))
        dr.text((x + TW + 4, y - 14), tb, fill=(0, 0, 0))
    out = os.path.join(OUTDIR, "compare.jpg")
    canvas.save(out, quality=88)
    print("\ncompare.jpg ->", out, canvas.size)
