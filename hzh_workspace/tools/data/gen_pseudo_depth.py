"""Generate pseudo depth labels for a YOLO detection dataset with a depth teacher model.

Writes uint16 PNG depth (millimeters, depth_scale=1000) to <root>/depth/<split>/<stem>.png, matching the
images/ -> depth/ layout that ultralytics.data.utils.load_depth and DepthDataset expect.

Usage:
    python hck_workspace/detdepth/gen_pseudo_depth.py --root /mnt/data2/VisDrone数据集 --splits train val test
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))  # this checkout's ultralytics, not a site-packages install

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.utils import IMG_FORMATS  # noqa: E402
from ultralytics.utils import TQDM  # noqa: E402


def main(opt):
    model = YOLO(opt.weights)
    root = Path(opt.root)
    for split in opt.splits:
        out_dir = root / "depth" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(f for f in (root / "images" / split).iterdir() if f.suffix[1:].lower() in IMG_FORMATS)
        todo = [str(f) for f in files if not (out_dir / f"{f.stem}.png").exists()]
        print(f"{split}: {len(files)} images, {len(todo)} to generate -> {out_dir}")
        for i in TQDM(range(0, len(todo), opt.batch), desc=split):
            for r in model.predict(todo[i : i + opt.batch], imgsz=opt.imgsz, device=opt.device, verbose=False):
                d = r.depth.data.float().cpu().numpy()  # (H, W) meters at original resolution
                mm = np.clip(d * 1000.0, 1, 65535).astype(np.uint16)  # >0 so every pixel counts as valid
                cv2.imwrite(str(out_dir / f"{Path(r.path).stem}.png"), mm)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/mnt/data2/COCO_PF")
    p.add_argument("--weights", default=str(REPO / "model_common" / "yolo26n-depth.pt"))
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="0")
    main(p.parse_args())
