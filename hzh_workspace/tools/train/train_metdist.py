#!/home/kang/miniforge3/envs/yolo/bin/python
"""metdist training entry: deep log-depth distance towers + warm start.

Same three-layer override contract as train_detdist.py:
  DEFAULTS < configs/metdist_train.yaml < CLI (--key value / key=value).
Prints the effective settings table; --dry-run stops before importing ultralytics.

Differences vs v1 entry:
  * trainer/model/loss come from libs/metdist (v1 untouched);
  * labels_root defaults to <dataset_root>/labels_dist_own (own-domain distance
    supervision only; coco/fpv keep detection labels, distance masked to -1);
  * depth_weights: optional backbone init from a metric-depth checkpoint
    (REPO-relative; disabled by default — mixing it with a trained neck zeroes
    mAP in smoke, see DEFAULTS comment);
  * det_weights: neck/detect init from a detection checkpoint (WS-relative);
    empty string disables either source; dist towers always start fresh;
  * dist_lambda 3.0 / dist_huber_delta 0.2 (log space), SGD lr0 0.01, cos_lr,
    epochs 160, scale 0.1, erasing 0.2.

Usage
-----
  ./train_detdist_v2.sh start                          # detached
  tools/train/train_metdist.py --dry-run            # print settings only
  ./train_detdist_v2.sh start --config configs/detdist_v2_smoke.yaml
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve()          # .../hzh_workspace/tools/train/train_metdist.py
WS = HERE.parents[2]                     # .../hzh_workspace
REPO = HERE.parents[3]                   # .../ultralytics-depth
sys.path[:0] = [str(WS / "libs"), str(REPO)]

DEFAULTS = {
    "model": "configs/metdist_model.yaml",
    "data": "configs/metdist_data.yaml",
    "labels_root": "",                     # "" -> <dataset_root>/labels_dist_own
    "project": "runs",
    "name": "metdist_pf",
    "epochs": 160,
    "imgsz": 640,
    "batch": 96,
    "device": "0",
    "workers": 8,
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "warmup_epochs": 3.0,
    "patience": 40,
    "seed": 0,
    "deterministic": True,
    "cache": False,
    "cos_lr": True,
    "close_mosaic": 10,
    "translate": 0.1,
    "scale": 0.1,
    "fliplr": 0.5,
    "flipud": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "erasing": 0.2,
    "dist_lambda": 3.0,
    "dist_huber_delta": 0.2,
    "dist_tau": 0.5,                       # <0.5: 近场高估重罚(读数偏小), 0.5=对称
    "dist_near_weight": 1.0,               # 近场(<5m)损失加权
    "dist_rank": 0.0,                      # 图内序关系损失权重(coco/fpv 多样性, 尺度无关)
    "metric_own": 0,                       # 1: 米制回归只给 own 域(coco/fpv 只参与序损失)
    "dist_dropout": 0.0,                   # 距离塔函数式 dropout(推理关闭)
    "bgswap": 0.0,                         # 背景置换增强概率(破坏场景记忆)
    "dist_mid": 32,                        # distance tower hidden width
    # 冒烟实测: depth backbone + 现成 neck 混装会使 mAP 归零(特征失配, 小数据救不回);
    # 默认只用检测热启动。depth backbone 可选(--depth_weights model_common/yolo26n-depth.pt),
    # 适合从头训或大数据长训。
    "depth_weights": "",                   # REPO-relative, "" disables
    "det_weights": "runs/detdist_pf_cal/weights/best.pt",  # WS-relative, "" disables
    "exist_ok": False,
    "plots": True,
}


def _smart(v: str):
    try:
        return yaml.safe_load(v)
    except Exception:  # noqa: BLE001
        return v


def _parse(argv: list[str]) -> dict:
    out: dict = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            if "=" in a:
                k, v = a[2:].split("=", 1)
            else:
                if i + 1 >= len(argv):
                    raise SystemExit(f"missing value for {a}")
                k, v = a[2:], argv[i + 1]
                i += 1
        elif "=" in a:
            k, v = a.split("=", 1)
        else:
            raise SystemExit(f"unrecognised argument: {a!r} (use --key value or key=value)")
        out[k.replace("-", "_")] = _smart(v)
        i += 1
    return out


def _dataset_root() -> str:
    cfg = WS / "configs" / "detdist_dataset.yaml"
    if cfg.is_file():
        root = (yaml.safe_load(cfg.read_text()) or {}).get("dataset_root")
        if root:
            return str(root)
    return "/mnt/data2/hzh/PF_11.9"


def _resolve_weights(v: str) -> str:
    """Resolve a weights path against WS then REPO; '' stays '' (disabled)."""
    if not v:
        return ""
    p = Path(str(v))
    if p.is_absolute():
        return str(p)
    for base in (WS, REPO):
        if (base / p).is_file():
            return str(base / p)
    raise SystemExit(f"weights not found: {v} (looked in {WS} and {REPO})")


def _fmt_imgsz(v) -> str:
    if isinstance(v, (list, tuple)):
        return "x".join(str(int(x)) for x in v) + "  (H,W)"
    return f"{int(v)}  (square)"


def main() -> None:
    argv = list(sys.argv[1:])
    dry = "--dry-run" in argv
    argv = [a for a in argv if a != "--dry-run"]

    cfg_path = WS / "configs" / "detdist_v2_train.yaml"
    if "--config" in argv:
        j = argv.index("--config")
        cfg_path = Path(argv[j + 1])
        del argv[j : j + 2]

    src = {k: "DEFAULTS" for k in DEFAULTS}
    cfg = dict(DEFAULTS)
    if cfg_path.is_file():
        for k, v in (yaml.safe_load(cfg_path.read_text()) or {}).items():
            cfg[k] = v
            src[k] = cfg_path.name
    else:
        print(f"[metdist] WARN config not found: {cfg_path} -> script DEFAULTS only", flush=True)

    for k, v in _parse(argv).items():
        cfg[k] = v
        src[k] = "CLI"

    os.environ["DETDIST_V2_LAMBDA"] = str(cfg.get("dist_lambda", 3.0))
    os.environ["DETDIST_V2_HUBER_DELTA"] = str(cfg.get("dist_huber_delta", 0.2))
    os.environ["DETDIST_V2_TAU"] = str(cfg.get("dist_tau", 0.5))
    os.environ["DETDIST_V2_NEAR_WEIGHT"] = str(cfg.get("dist_near_weight", 1.0))
    os.environ["DETDIST_V2_RANK"] = str(cfg.get("dist_rank", 0.0))
    os.environ["DETDIST_V2_METRIC_OWN"] = str(cfg.get("metric_own", 0))
    os.environ["DETDIST_V2_DROPOUT"] = str(cfg.get("dist_dropout", 0.0))
    os.environ["DETDIST_V2_BGSWAP"] = str(cfg.get("bgswap", 0.0))
    os.environ["DETDIST_V2_MID"] = str(cfg.get("dist_mid", 32))
    labels_root = str(
        cfg.get("labels_root")
        or os.environ.get("DETDIST_LABELS_ROOT")
        or (Path(_dataset_root()) / "labels_dist_own")
    )
    os.environ["DETDIST_LABELS_ROOT"] = labels_root
    if not cfg.get("labels_root"):
        cfg["labels_root"] = labels_root
        src["labels_root"] = "derived (<dataset_root>/labels_dist_own)"

    depth_weights = _resolve_weights(str(cfg.get("depth_weights") or ""))
    det_weights = _resolve_weights(str(cfg.get("det_weights") or ""))

    def _abs(key: str) -> Path:
        p = Path(str(cfg[key]))
        return p if p.is_absolute() else (WS / p)

    model_yaml = _abs("model")
    data_yaml = _abs("data")
    project = _abs("project")
    run_dir = project / str(cfg["name"])
    imgsz = tuple(cfg["imgsz"]) if isinstance(cfg["imgsz"], (list, tuple)) else int(cfg["imgsz"])

    show = [
        "model", "data", "labels_root", "project", "name", "epochs", "imgsz", "batch",
        "device", "workers", "optimizer", "lr0", "lrf", "warmup_epochs", "patience",
        "seed", "deterministic", "cache", "cos_lr", "close_mosaic", "translate", "scale",
        "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v", "erasing",
        "dist_lambda", "dist_huber_delta", "dist_tau", "dist_near_weight",
        "dist_rank", "metric_own", "dist_dropout", "bgswap",
        "dist_mid", "depth_weights", "det_weights",
        "exist_ok", "plots",
    ]
    print("=" * 100, flush=True)
    print("[metdist] effective training settings   (source: DEFAULTS < yaml < CLI)", flush=True)
    print("-" * 100, flush=True)
    for k in show:
        v = cfg.get(k, DEFAULTS.get(k))
        if k == "imgsz":
            v = _fmt_imgsz(v)
        if k == "depth_weights":
            v = depth_weights or "(disabled)"
        if k == "det_weights":
            v = det_weights or "(disabled)"
        print(f"  {k:<16}: {str(v):<52} <- {src.get(k, 'DEFAULTS')}", flush=True)
    print("-" * 100, flush=True)
    print(f"  {'model (abs)':<16}: {model_yaml}", flush=True)
    print(f"  {'data (abs)':<16}: {data_yaml}", flush=True)
    print(f"  {'run dir':<16}: {run_dir}   (project={project}, name={cfg['name']})", flush=True)
    print(f"  {'imgsz meaning':<16}: H x W used for train+val; ONNX export must use the same size", flush=True)
    print("=" * 100, flush=True)

    if dry:
        print("[metdist] --dry-run: 只打印参数，不加载模型、不训练", flush=True)
        return

    from ultralytics import YOLO  # noqa: E402

    from metdist.trainer import DistV2Trainer  # noqa: E402

    DistV2Trainer.labels_root = labels_root
    DistV2Trainer.depth_weights = depth_weights
    DistV2Trainer.det_weights = det_weights

    train_args = dict(
        data=str(data_yaml),
        epochs=int(cfg["epochs"]),
        imgsz=imgsz,
        batch=int(cfg["batch"]),
        device=str(cfg["device"]),
        workers=int(cfg["workers"]),
        optimizer=str(cfg["optimizer"]),
        lr0=float(cfg["lr0"]),
        lrf=float(cfg["lrf"]),
        warmup_epochs=float(cfg["warmup_epochs"]),
        patience=int(cfg["patience"]),
        seed=int(cfg["seed"]),
        deterministic=bool(cfg["deterministic"]),
        cache=cfg["cache"],
        cos_lr=bool(cfg["cos_lr"]),
        close_mosaic=int(cfg["close_mosaic"]),
        translate=float(cfg["translate"]),
        scale=float(cfg["scale"]),
        fliplr=float(cfg["fliplr"]),
        flipud=float(cfg["flipud"]),
        hsv_h=float(cfg["hsv_h"]),
        hsv_s=float(cfg["hsv_s"]),
        hsv_v=float(cfg["hsv_v"]),
        erasing=float(cfg["erasing"]),
        project=str(project),
        name=str(cfg["name"]),
        exist_ok=bool(cfg.get("exist_ok", False)),
        plots=bool(cfg.get("plots", True)),
        val=True,
    )
    for k, v in cfg.items():
        if k not in train_args and k not in {
            "model", "data", "labels_root", "dist_lambda", "dist_huber_delta",
            "dist_tau", "dist_near_weight", "dist_rank", "metric_own",
            "dist_dropout", "bgswap",
            "dist_mid", "depth_weights", "det_weights", "project", "name",
        }:
            train_args[k] = v

    YOLO(str(model_yaml)).train(trainer=DistV2Trainer, **train_args)


if __name__ == "__main__":
    main()