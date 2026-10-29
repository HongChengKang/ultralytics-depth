"""将固定 batch 的 split ONNX（export_onnx_split.py）转为 RKNN，可选 int8 量化，并做精度检查。

量化校准（仅 int8，不加 --fp 时）:
  - 源图目录默认: /mnt/data2/PFV_dataset/images/train（可用 --dataset 覆盖）
  - 等间隔抽 --num 张，与部署一致做 letterbox（灰边 114），再按 batch 打成 NCHW uint8 .npy
  - 固定 batch-N 模型每个校准样本必须是 N 张图；产物写在 <out.rknn 同名>.calib/
    （dataset.txt + calib_XXXX.npy）

精度检查（--check N）:
  - 默认用 /mnt/data2/PFV_dataset/images/val 抽 N 张
  - RKNN PC 模拟器 vs onnxruntime，对比检测框与框内深度

须在 rknn-toolkit2 环境运行:
    /home/eav-nanjin/miniforge3/envs/rknn/bin/python hck_workspace/detdepth/convert_rknn.py \\
        <model.onnx> <out.rknn> --dataset /mnt/data2/PFV_dataset/images/train --num 300 --check 40 [--fp]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx
from rknn.api import RKNN

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rknn_infer import IMG_EXTS, Runner, letterbox, postprocess, preprocess  # noqa: E402


def build_calibration(src: Path, num: int, batch: int, size: int, work: Path) -> Path:
    """Write evenly sampled, letterboxed RGB batches as NCHW uint8 .npy and return the dataset.txt listing them."""
    txt = work / "dataset.txt"
    files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMG_EXTS)
    num = min(num, len(files)) // batch * batch
    if txt.exists() and len(txt.read_text().split()) == num // batch:
        print(f"calibration: reusing {txt}")
        return txt
    picks = [files[int(i * len(files) / num)] for i in range(num)]
    work.mkdir(parents=True, exist_ok=True)
    lines = []
    for k in range(0, num, batch):
        rgb = [letterbox(cv2.imread(str(f)), size)[0][:, :, ::-1] for f in picks[k : k + batch]]
        npy = work / f"calib_{k // batch:04d}.npy"
        np.save(npy, np.ascontiguousarray(np.stack(rgb).transpose(0, 3, 1, 2)))
        lines.append(str(npy.resolve()))  # rknn resolves relative entries against the txt's directory
    txt.write_text("\n".join(lines) + "\n")
    print(f"calibration: {num} images from {src} -> {len(lines)} batches of {batch} in {work}")
    return txt


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), 2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], 1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def check(rknn: RKNN, onnx_path: str, src: Path, n: int, batch: int, size: int) -> None:
    """Compare simulator detections/box depths with onnxruntime on n images (IoU>=0.5, same class = match)."""
    ref = Runner(onnx_path)
    files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMG_EXTS)
    files = [files[int(i * len(files) / n)] for i in range(n)]
    n_ref = n_sim = n_match = 0
    depth_err, conf_err, map_diff = [], [], []
    for k in range(0, n, batch):
        ims = [cv2.imread(str(f)) for f in files[k : k + batch]]
        x, metas = preprocess(ims, size, batch)
        o_ref = ref(x)
        sim = [o.astype(np.float32) for o in rknn.inference(inputs=[x], data_format="nhwc")]
        r_onnx, r_sim = postprocess(o_ref, metas), postprocess(sim, metas)
        map_diff.append(np.abs(sim[9] - o_ref[9]).mean() / o_ref[9].mean())
        for a, b in zip(r_onnx, r_sim):
            n_ref, n_sim = n_ref + len(a["xyxy"]), n_sim + len(b["xyxy"])
            if not len(a["xyxy"]) or not len(b["xyxy"]):
                continue
            m = iou_matrix(a["xyxy"], b["xyxy"]) * (a["cls"][:, None] == b["cls"][None])
            for i in range(len(a["xyxy"])):
                j = m[i].argmax()
                if m[i, j] >= 0.5:
                    n_match += 1
                    depth_err.append(abs(b["depth"][j] - a["depth"][i]) / a["depth"][i])
                    conf_err.append(abs(b["conf"][j] - a["conf"][i]))
                    m[:, j] = 0
    print(f"\n==== accuracy check on {n} images from {src} (onnx = reference) ====")
    print(f"boxes: onnx {n_ref}, rknn {n_sim}, matched {n_match}")
    print(f"recall vs onnx {n_match / max(n_ref, 1):.1%}, precision vs onnx {n_match / max(n_sim, 1):.1%}")
    if depth_err:
        print(f"box depth rel. error: median {np.median(depth_err):.2%}, p90 {np.percentile(depth_err, 90):.2%}")
        print(f"conf abs. error: median {np.median(conf_err):.3f}")
    print(f"depth map mean rel. error: {np.mean(map_diff):.2%}")


def main(opt):
    dims = [d.dim_value for d in onnx.load(opt.onnx).graph.input[0].type.tensor_type.shape.dim]
    batch, size = dims[0], dims[2]
    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=opt.platform)
    assert rknn.load_onnx(model=opt.onnx) == 0, "load_onnx failed"
    dataset = None
    if not opt.fp:
        dataset = build_calibration(Path(opt.dataset), opt.num, batch, size, Path(opt.out).with_suffix(".calib"))
    assert rknn.build(do_quantization=not opt.fp, dataset=str(dataset) if dataset else None) == 0, "build failed"
    assert rknn.export_rknn(opt.out) == 0, "export failed"
    print(f"saved {opt.out} ({'fp16' if opt.fp else 'int8'}, {opt.platform}, batch {batch}, imgsz {size})")
    if opt.check:
        assert rknn.init_runtime() == 0, "simulator init failed"
        check(rknn, opt.onnx, Path(opt.val), opt.check, batch, size)
    rknn.release()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("onnx")
    p.add_argument("out")
    p.add_argument("--dataset", default="/mnt/data2/PFV_dataset/images/train", help="calibration images")
    p.add_argument("--num", type=int, default=300, help="number of calibration images")
    p.add_argument("--platform", default="rk3576")
    p.add_argument("--fp", action="store_true", help="no quantization (fp16)")
    p.add_argument("--check", type=int, default=0, help="compare N images against onnxruntime on the PC simulator")
    p.add_argument("--val", default="/mnt/data2/PFV_dataset/images/val", help="images for --check")
    main(p.parse_args())
