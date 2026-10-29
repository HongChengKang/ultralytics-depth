#!/usr/bin/env bash
# PFV_dataset YOLO26-DetDepth：检测从 yaml 起、深度头从 yolo26n-depth.pt 迁移。
#
#   ./train_pfv_dataset.sh
#   ./train_pfv_dataset.sh epochs=2 batch=32
#
# 数据集：/mnt/data2/PFV_dataset（train 8559 / val 945）
# 入口：tools/train/train_detdepth_pfv.py
# 默认 run 名：runs/detdepth_pfv_dataset
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P="$ROOT/hzh_workspace"
PY=/home/kang/miniforge3/envs/yolo/bin/python

export PYTHONPATH="$P/env/pylibs:$ROOT/hck_workspace/detdepth:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_CONFIG_DIR="$P/env/cfg"
export MPLCONFIGDIR="$P/env/cfg/mpl"
export TORCH_HOME="$P/env/cfg/torch"
mkdir -p "$P/env/cfg/mpl" "$P/env/cfg/torch"

exec "$PY" "$P/tools/train/train_detdepth_pfv.py" "$@"
