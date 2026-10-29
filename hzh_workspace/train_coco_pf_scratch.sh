#!/usr/bin/env bash
# COCO_PF YOLO26-DetDepth：检测 + 深度头都从零（depth_weights=None，不用 teacher）。
#
#   ./train_coco_pf_scratch.sh
#   ./train_coco_pf_scratch.sh epochs=2 batch=8
#
# 注意：这条线只用于“纯从零”对照/冒烟，不是 COCO_PF v1/v2 的实际配方；
# v1/v2 走的是 ./train_coco_pf.sh（检测从 yaml 起、深度头从 teacher 迁移）。
# 入口：tools/train/train_detdepth_from_scratch.py
# 默认 run 名：runs/detdepth_coco_pf_scratch
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P="$ROOT/hzh_workspace"
PY=/home/kang/miniforge3/envs/yolo/bin/python

export PYTHONPATH="$P/env/pylibs:$ROOT/hck_workspace/detdepth:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_CONFIG_DIR="$P/env/cfg"
export MPLCONFIGDIR="$P/env/cfg/mpl"
export TORCH_HOME="$P/env/cfg/torch"
mkdir -p "$P/env/cfg/mpl" "$P/env/cfg/torch"

exec "$PY" "$P/tools/train/train_detdepth_from_scratch.py" "$@"
