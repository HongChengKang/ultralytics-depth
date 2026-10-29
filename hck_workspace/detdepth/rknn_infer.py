"""Inference + post-processing for the split detdepth export (export_onnx_split.py): boxes with per-box depth.

- PC: use .onnx (onnxruntime). Passing .rknn without rknn-toolkit-lite2 auto-falls back to a sibling .onnx
  (same stem with `_int8`/`_fp16` stripped, e.g. `*_b1_int8.rknn` → `*_b1.onnx`).
- Board (rk3576/rk3588): .rknn via rknn-toolkit-lite2; pass `--batch 1` for b1 models.

    python rknn_infer.py --model pf_detdepth_from_scratch_v2_split_b1.onnx --source images/ --out results/
    python rknn_infer.py --model pf_detdepth_from_scratch_v2_split_b1_int8.rknn --source images/ --out results/ --batch 1

Writes <out>/<stem>.jpg (boxes labeled "cls conf | d=<depth>") and <out>/<stem>.txt ("cls conf x1 y1 x2 y2 depth",
original-image pixels).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

STRIDES = (8, 16, 32)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _sibling_onnx(rknn_path: Path):
    """Resolve companion ONNX next to an RKNN (strip trailing _int8/_fp16 before .onnx)."""
    stem = rknn_path.stem
    for suffix in ("_int8", "_fp16"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    cand = rknn_path.with_name(stem + ".onnx")
    return cand if cand.is_file() else None


def letterbox(im: np.ndarray, size: int):
    """Resize keeping aspect ratio and pad to size x size with 114 (centered); return image, ratio, (left, top)."""
    h, w = im.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[top : top + nh, left : left + nw] = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, (left, top)


def preprocess(images: list, size: int, batch: int):
    """Letterbox BGR images to an RGB NHWC uint8 batch (padded to `batch`); return batch and per-image metas."""
    x = np.full((batch, size, size, 3), 114, np.uint8)
    metas = []
    for i, im in enumerate(images):
        lb, r, pad = letterbox(im, size)
        x[i] = lb[:, :, ::-1]
        metas.append((r, pad, im.shape[:2]))
    return x, metas


def postprocess(outs: list, metas: list, conf: float = 0.25, iou: float = 0.7, max_det: int = 300) -> list:
    """Decode per-level box/cls/score_sum outputs, run class-aware NMS, and take each box's median depth.

    Args:
        outs (list[np.ndarray]): 10 outputs in export order (box, cls, score_sum per P3/P4/P5, then depth), NCHW.
        metas (list[tuple]): (ratio, (left, top), (orig_h, orig_w)) per real image in the batch.

    Returns:
        (list[dict]): Per image, "xyxy" (N, 4) original-image pixels, "conf" (N,), "cls" (N,), "depth" (N,).
    """
    depth = outs[9]
    size = depth.shape[-1]
    results = []
    for b, (r, (left, top), (h0, w0)) in enumerate(metas):
        boxes, scores, classes = [], [], []
        for li, s in enumerate(STRIDES):
            box, cls, ssum = outs[3 * li][b], outs[3 * li + 1][b], outs[3 * li + 2][b, 0]
            ys, xs = np.nonzero(ssum >= conf)  # cheap pre-filter: max class score <= score sum
            if not len(ys):
                continue
            sc = cls[:, ys, xs]
            c, m = sc.argmax(0), sc.max(0)
            k = m >= conf
            ys, xs, c, m = ys[k], xs[k], c[k], m[k]
            ltrb = box[:, ys, xs] * s
            cx, cy = (xs + 0.5) * s, (ys + 0.5) * s
            boxes.append(np.stack([cx - ltrb[0], cy - ltrb[1], cx + ltrb[2], cy + ltrb[3]], 1))
            scores.append(m)
            classes.append(c)
        if not boxes:
            results.append({"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "cls": np.zeros(0, int), "depth": np.zeros(0)})
            continue
        boxes, scores, classes = np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes)
        off = boxes + classes[:, None] * 4096.0  # class-aware NMS by offsetting each class into its own region
        keep = cv2.dnn.NMSBoxes(
            np.c_[off[:, :2], off[:, 2:] - off[:, :2]].tolist(), scores.tolist(), conf, iou, top_k=max_det
        )
        keep = np.array(keep, int).reshape(-1)
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        d = np.array([box_depth(depth[b, 0], bx, size) for bx in boxes])
        xyxy = (boxes - [left, top, left, top]) / r  # undo letterbox
        xyxy = np.clip(xyxy, 0, [w0, h0, w0, h0])
        results.append({"xyxy": xyxy, "conf": scores, "cls": classes, "depth": d})
    return results


def box_depth(depth: np.ndarray, xyxy: np.ndarray, size: int) -> float:
    """Median depth inside a letterbox-space box; the clamped center pixel if the box is degenerate."""
    x1, y1, x2, y2 = np.clip(np.round(xyxy).astype(int), 0, size)
    if x2 > x1 and y2 > y1:
        return float(np.median(depth[y1:y2, x1:x2]))
    return float(depth[min((y1 + y2) // 2, size - 1), min((x1 + x2) // 2, size - 1)])


class Runner:
    """Run the model on an RGB NHWC uint8 batch with onnxruntime (.onnx) or rknn-toolkit-lite2 (.rknn)."""

    def __init__(self, model: str):
        """Load the model and read its fixed batch size and input size."""
        path = Path(model)
        self.rknn = None
        self.sess = None
        if path.suffix.lower() == ".onnx":
            self._load_onnx(path)
            return
        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            onnx = _sibling_onnx(path)
            if onnx is None:
                raise SystemExit(
                    f"PC 无 rknnlite，无法直接跑 .rknn（{path}）。\n"
                    f"请改用配套 .onnx，例如：--model {path.with_name(path.stem.replace('_int8','').replace('_fp16','') + '.onnx')}\n"
                    f"板端需安装 rknn-toolkit-lite2 后再传 .rknn。"
                ) from None
            print(f"[warn] PC 无 rknnlite，改用配套 ONNX: {onnx}")
            self._load_onnx(onnx)
            return

        self.rknn = RKNNLite()
        assert self.rknn.load_rknn(str(path)) == 0, f"load_rknn failed: {path}"
        # rk3576 为双核 NPU（0+1）；rk3588 才是 0_1_2
        core = getattr(RKNNLite, "NPU_CORE_0_1", getattr(RKNNLite, "NPU_CORE_AUTO", 0))
        assert self.rknn.init_runtime(core_mask=core) == 0, "init_runtime failed"
        # b1 模型请传 --batch 1；默认按常见 b4 导出
        self.batch, self.size = 4, 640

    def _load_onnx(self, path: Path):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.batch, _, self.size, _ = self.sess.get_inputs()[0].shape
        self.rknn = None

    def __call__(self, x: np.ndarray) -> list:
        """Return the 10 outputs as float32 NCHW arrays."""
        if self.rknn is None:
            return self.sess.run(None, {"images": x.transpose(0, 3, 1, 2).astype(np.float32) / 255})
        return [o.astype(np.float32) for o in self.rknn.inference(inputs=[x], data_format="nhwc")]


def draw(im: np.ndarray, res: dict, names: list) -> np.ndarray:
    """Draw boxes labeled with class, confidence and depth."""
    for (x1, y1, x2, y2), c, s, d in zip(res["xyxy"].astype(int), res["cls"], res["conf"], res["depth"]):
        color = (56, 56, 255) if c == 0 else (255, 157, 151)
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
        label = f"{names[c] if c < len(names) else c} {s:.2f} | d={d:.1f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(im, (x1, max(y1 - th - 4, 0)), (x1 + tw, max(y1, th + 4)), color, -1)
        cv2.putText(im, label, (x1, max(y1 - 3, th)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return im


def main(opt):
    run = Runner(opt.model)
    if opt.batch:
        run.batch = opt.batch
    if opt.imgsz:
        run.size = opt.imgsz
    src = Path(opt.source)
    files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMG_EXTS) if src.is_dir() else [src]
    out = Path(opt.out)
    out.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(files), run.batch):
        chunk = files[k : k + run.batch]
        ims = [cv2.imread(str(f)) for f in chunk]
        x, metas = preprocess(ims, run.size, run.batch)
        for f, im, res in zip(chunk, ims, postprocess(run(x), metas, opt.conf, opt.iou)):
            cv2.imwrite(str(out / f"{f.stem}.jpg"), draw(im, res, opt.names))
            rows = zip(res["cls"], res["conf"], res["xyxy"], res["depth"])
            lines = [f"{c} {s:.4f} {b[0]:.1f} {b[1]:.1f} {b[2]:.1f} {b[3]:.1f} {d:.3f}\n" for c, s, b, d in rows]
            (out / f"{f.stem}.txt").write_text("".join(lines))
            print(f"{f.name}: {len(lines)} boxes")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help=".onnx (PC) or .rknn (board)")
    p.add_argument("--source", required=True, help="image file or directory")
    p.add_argument("--out", default="rknn_results")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--batch", type=int, default=0, help="override the model batch size (rknn)")
    p.add_argument("--imgsz", type=int, default=0, help="override the model input size (rknn)")
    p.add_argument("--names", nargs="+", default=["person", "Fertilizer bag"])
    main(p.parse_args())
