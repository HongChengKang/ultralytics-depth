"""Inference + post-processing for the split detdepth / dist-plan-a export (export_onnx_split.py).

Supports both export flavours:
  * 10 outputs (detdepth): 9 detection tensors + 1 depth map -> box depth = median of the depth map
  * 12 outputs (detdist):   9 detection tensors + 3 metric distance maps (metres) -> box distance = median of
                           the anchors of all three levels that fall inside the box

Depends only on numpy + OpenCV, so it runs on the PC (.onnx via onnxruntime) and on the RK board (.rknn via
rknn-toolkit-lite2):
    python rknn_infer.py --model pfv_detdepth_split_int8.rknn --source images/ --out results/
    python rknn_infer.py --model pfv_detdepth_split.onnx --source images/ --out results/

Writes <out>/<stem>.jpg (boxes labeled "cls conf | d=<depth>") and <out>/<stem>.txt ("cls conf x1 y1 x2 y2 depth",
original-image pixels).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

STRIDES = (8, 16, 32)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def as_hw(size) -> tuple:
    """Normalise an input size to (h, w): 640 -> (640, 640), "480x320" -> (480, 320), (480, 320) -> unchanged."""
    if isinstance(size, (tuple, list)):
        return int(size[0]), int(size[1])
    parts = str(size).lower().replace(",", "x").replace(" ", "").split("x")
    h = int(parts[0])
    return h, (int(parts[1]) if len(parts) > 1 else h)


def letterbox(im: np.ndarray, size):
    """Resize keeping aspect ratio and pad to hw=(h, w) with 114 (centered); return image, ratio, (left, top)."""
    h, w = as_hw(size)
    h0, w0 = im.shape[:2]
    r = min(h / h0, w / w0)
    nh, nw = round(h0 * r), round(w0 * r)
    top, left = (h - nh) // 2, (w - nw) // 2
    out = np.full((h, w, 3), 114, np.uint8)
    out[top : top + nh, left : left + nw] = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, (left, top)


def preprocess(images: list, size, batch: int):
    """Letterbox BGR images to an RGB NHWC uint8 batch of hw=(h, w) (padded to `batch`); return batch and metas."""
    h, w = as_hw(size)
    x = np.full((batch, h, w, 3), 114, np.uint8)
    metas = []
    for i, im in enumerate(images):
        lb, r, pad = letterbox(im, (h, w))
        x[i] = lb[:, :, ::-1]
        metas.append((r, pad, im.shape[:2]))
    return x, metas


def _warn_if_quantized(outs: list) -> None:
    """int8/int8 量化模型的输出可能是整型（rknn 会把 dist_p* 的 output dtype 改成 int8）。

    这种输出必须用运行时给的 zp/scale 反量化后才等于米；fp16 模型没有这个问题。
    """
    bad = [i for i, o in enumerate(outs) if np.asarray(o).dtype in (np.int8, np.uint8, np.int16)]
    if bad:
        print(f"[warn] outputs {bad} are integer-typed (int8 quantization): dequantize with the runtime's "
              f"zp/scale before using them as metres, or deploy the fp16 model")


def postprocess(outs: list, metas: list, conf: float = 0.25, iou: float = 0.7, max_det: int = 300) -> list:
    """Decode per-level box/cls/score_sum outputs, run class-aware NMS, and take each box's median depth.

    Args:
        outs (list[np.ndarray]): 10 or 12 outputs in export order (box, cls, score_sum per P3/P4/P5, then
            either one depth map or one distance map per level), NCHW.
        metas (list[tuple]): (ratio, (left, top), (orig_h, orig_w)) per real image in the batch.

    Returns:
        (list[dict]): Per image, "xyxy" (N, 4) original-image pixels, "conf" (N,), "cls" (N,), "depth" (N,).
    """
    n_out = len(outs)
    dist_maps = outs[9:12] if n_out >= 12 else None
    if dist_maps is not None:
        dist_maps = [np.asarray(m, np.float32) for m in dist_maps]
        if np.asarray(outs[9]).dtype in (np.int8, np.uint8, np.int16):
            _warn_if_quantized(outs)   # detdist: dist_p3/p4/p5, 已是米
    depth = outs[9] if n_out < 12 else None           # 旧 detdepth: 单张深度图
    h, w = as_hw(dist_maps[0].shape[-2:] if dist_maps is not None else depth.shape[-2:])
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
        if dist_maps is not None:
            d = np.array([box_dist_levels([m[b, 0] for m in dist_maps], bx, (h, w)) for bx in boxes])
        else:
            d = np.array([box_depth(depth[b, 0], bx, (h, w)) for bx in boxes])
        xyxy = (boxes - [left, top, left, top]) / r  # undo letterbox
        xyxy = np.clip(xyxy, 0, [w0, h0, w0, h0])
        results.append({"xyxy": xyxy, "conf": scores, "cls": classes, "depth": d})
    return results


def box_dist_levels(dmap: list, xyxy: np.ndarray, size) -> float:
    """Median metric distance (metres) over the P3/P4/P5 anchors whose centres fall inside a letterbox box.

    Args:
        dmap (list[np.ndarray]): three (H_s, W_s) distance maps for strides 8/16/32, already in metres.
        xyxy (np.ndarray): box in letterbox pixels.
        size: input size (h, w).

    Returns:
        float: median distance in metres of all anchors inside the box (0.0 if the box is empty).
    """
    h, w = as_hw(size)
    x1, y1, x2, y2 = np.clip(xyxy, 0, [w, h, w, h])
    vals = []
    for s, m in zip(STRIDES, dmap):
        gh, gw = m.shape[-2:]
        j0 = max(0, min(int(np.floor(x1 / s - 0.5)), gw - 1))
        j1 = max(j0 + 1, min(int(np.ceil(x2 / s - 0.5)), gw))
        i0 = max(0, min(int(np.floor(y1 / s - 0.5)), gh - 1))
        i1 = max(i0 + 1, min(int(np.ceil(y2 / s - 0.5)), gh))
        vals.append(m[i0:i1, j0:j1].ravel())
    v = np.concatenate(vals)
    return float(np.median(v)) if v.size else 0.0


def box_depth(depth: np.ndarray, xyxy: np.ndarray, size) -> float:
    """Median depth inside a letterbox-space box (clipped to the hw input size); center pixel if degenerate."""
    h, w = as_hw(size)
    x1, y1, x2, y2 = np.clip(np.round(xyxy).astype(int), [0, 0, 0, 0], [w, h, w, h])
    if x2 > x1 and y2 > y1:
        return float(np.median(depth[y1:y2, x1:x2]))
    return float(depth[min((y1 + y2) // 2, h - 1), min((x1 + x2) // 2, w - 1)])


class Runner:
    """Run the model on an RGB NHWC uint8 batch with onnxruntime (.onnx) or rknn-toolkit-lite2 (.rknn)."""

    def __init__(self, model: str):
        """Load the model and read its fixed batch size and input size (h, w)."""
        if model.endswith(".onnx"):
            import onnxruntime as ort

            self.sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
            self.batch, _, h, w = self.sess.get_inputs()[0].shape
            self.hw = as_hw((h, w))
            self.rknn = None
        else:
            from rknnlite.api import RKNNLite

            self.rknn = RKNNLite()
            assert self.rknn.load_rknn(model) == 0, f"load_rknn failed: {model}"
            # rk3576 为双核 NPU（0+1）；rk3588 才是 0_1_2
            core = getattr(RKNNLite, "NPU_CORE_0_1", getattr(RKNNLite, "NPU_CORE_AUTO", 0))
            assert self.rknn.init_runtime(core_mask=core) == 0, "init_runtime failed"
            self.batch, self.hw = 4, (640, 640)  # export_onnx_split.py defaults; override with --batch/--imgsz

    def __call__(self, x: np.ndarray) -> list:
        """Return the 10 (detdepth) or 12 (detdist) outputs as float32 NCHW arrays."""
        if self.rknn is None:
            return self.sess.run(None, {"images": x.transpose(0, 3, 1, 2).astype(np.float32) / 255})
        return [o.astype(np.float32) for o in self.rknn.inference(inputs=[x], data_format="nhwc")]


def draw(im: np.ndarray, res: dict, names: list) -> np.ndarray:
    """Draw boxes labeled with class, confidence and depth."""
    for (x1, y1, x2, y2), c, s, d in zip(res["xyxy"].astype(int), res["cls"], res["conf"], res["depth"]):
        color = (56, 56, 255) if c == 0 else (255, 157, 151)
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
        label = f"{names[c] if c < len(names) else c} {s:.2f} | d={d:.2f}m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(im, (x1, max(y1 - th - 4, 0)), (x1 + tw, max(y1, th + 4)), color, -1)
        cv2.putText(im, label, (x1, max(y1 - 3, th)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return im


def main(opt):
    run = Runner(opt.model)
    if opt.batch:
        run.batch = opt.batch
    if opt.imgsz:
        run.hw = as_hw(opt.imgsz)
    src = Path(opt.source)
    files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMG_EXTS) if src.is_dir() else [src]
    out = Path(opt.out)
    out.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(files), run.batch):
        chunk = files[k : k + run.batch]
        ims = [cv2.imread(str(f)) for f in chunk]
        x, metas = preprocess(ims, run.hw, run.batch)
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
    p.add_argument("--imgsz", default="", help="override the input size (.rknn 读不到 shape): 640 或 高x宽，如 480x320")
    p.add_argument("--names", nargs="+", default=["person", "Fertilizer bag"])
    main(p.parse_args())
