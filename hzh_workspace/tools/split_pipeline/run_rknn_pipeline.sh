#!/usr/bin/env bash
# =============================================================================
# detdepth：.pt → split ONNX → fp16 / int8 RKNN（rk3576）完整流水线
#
# 用法:
#   bash hzh_workspace/tools/split_pipeline/run_rknn_pipeline.sh [best.pt] [out_dir]
#
# 默认（v2 正式 run）:
#   best.pt  = hzh_workspace/runs/detdepth_coco_pf/weights/best.pt
#   out_dir  = hzh_workspace/models/split_pf_v2
# v1 旧 run 需显式指定: runs/detdepth_coco_pf_old/weights/best.pt + models/split_pf
#
# 环境依赖（两套 Python，勿混用）:
#   YOLO_PY  — 含 torch + 本仓库 ultralytics，负责 .pt → ONNX
#   RKNN_PY  — rknn-toolkit2（如 2.3.2），负责 ONNX → RKNN、精度检查、ONNX 预览
#
# 数据路径（量化 / 校验）:
#   校准图（int8）: /mnt/data2/PFV_dataset/images/train
#   校验图（--check / 预览）: /mnt/data2/PFV_dataset/images/val
#   校准产物目录:  <out_dir>/<NAME>_split_int8.calib/
#                 （内含 calib_XXXX.npy + dataset.txt，每份 npy 为 batch=N 的 NCHW uint8）
#
# 流程概览:
#   1) export_onnx_split.py  — 导出固定 batch=N（默认 1）的多头 ONNX
#                              输出: box/cls/score_sum × P3/P4/P5 + depth
#   2) convert_rknn.py --fp  — 无量化 fp16 RKNN + PC 模拟器 vs ONNX 对比
#   3) convert_rknn.py       — int8 量化 RKNN（用 train 校准图）+ 同上精度检查
#   4) rknn_infer.py         — 用 ONNX 对少量 val 图出预览，作部署对照
#
# 板端推理（rk3576，需 rknn-toolkit-lite2 + opencv + numpy）:
#   拷贝 rknn_infer.py 与 *.rknn 后:
#     python3 rknn_infer.py --model <NAME>_split_int8.rknn --source <images_dir> --out results
# =============================================================================
set -euo pipefail

REPO=/home/kang/workspace/projects/yolo-depth/ultralytics-depth
PT=${1:-$REPO/hzh_workspace/runs/detdepth_coco_pf/weights/best.pt}  # v2 正式；v1 用 ..._old
OUT=${2:-$REPO/hzh_workspace/models/split_pf_v2}
YOLO_PY=/home/kang/miniforge3/envs/yolo/bin/python       # torch + 本仓库 ultralytics
RKNN_PY=/home/eav-nanjin/miniforge3/envs/rknn/bin/python # rknn-toolkit2 2.3.2
D=$REPO/hzh_workspace/tools/split_pipeline
# 用 run 目录名当产物前缀，例如 detdepth_pfv_dataset
NAME=$(basename "$(dirname "$(dirname "$PT")")")

BATCH=${BATCH:-1}   # 模型固定 batch，默认 1

# 目标芯片：rk3576（勿用 rk3588，板端会加载失败）
PLATFORM=${PLATFORM:-rk3588}

# 量化校准图目录（convert_rknn.py 默认值，此处显式写出便于改路径）
CALIB_DIR=/mnt/data2/COCO_PF/images/train
# 精度检查 / 预览用验证集
VAL_DIR=/mnt/data2/COCO_PF/images/val

echo "== 1/4 导出 split ONNX（batch=$BATCH，多头 box/cls/score_sum + depth）"
# 产物: $OUT/${NAME}_split_b${BATCH}.onnx（与芯片无关，可复用）
"$YOLO_PY" "$D/export_onnx_split.py" "$PT" "$OUT/${NAME}_split_b${BATCH}.onnx" --batch $BATCH

echo "== 2/4 转 fp16 RKNN（$PLATFORM，不做量化）+ PC 模拟器精度检查（40 张 val）"
# 产物: $OUT/${NAME}_split_b${BATCH}_fp16.rknn
"$RKNN_PY" "$D/convert_rknn.py" "$OUT/${NAME}_split_b${BATCH}.onnx" "$OUT/${NAME}_split_b${BATCH}_fp16.rknn" \
    --platform "$PLATFORM" --fp --check 40 --val "$VAL_DIR"

echo "== 3/4 转 int8 RKNN（$PLATFORM，PTQ，校准图=$CALIB_DIR，抽样 300 张）+ 精度检查（40 张 val）"
# 校准: 等间隔抽 --num 张，letterbox 灰边 114，按 batch=$BATCH 打成 NCHW uint8 .npy
# 校准工作目录: $OUT/${NAME}_split_b${BATCH}_int8.calib/（dataset.txt 列出各 calib_XXXX.npy）
# 产物: $OUT/${NAME}_split_b${BATCH}_int8.rknn
"$RKNN_PY" "$D/convert_rknn.py" "$OUT/${NAME}_split_b${BATCH}.onnx" "$OUT/${NAME}_split_b${BATCH}_int8.rknn" \
    --platform "$PLATFORM" --dataset "$CALIB_DIR" --num 300 --check 40 --val "$VAL_DIR"

echo "== 4/4 ONNX 参考预览（从 val 抽 8 张）-> $OUT/preview_onnx"
mkdir -p "$OUT/preview_src"
find "$VAL_DIR" -maxdepth 1 -type f | sort | awk 'NR % 100 == 1' | head -8 |
    xargs -I{} ln -sf {} "$OUT/preview_src/"
"$RKNN_PY" "$D/rknn_infer.py" --model "$OUT/${NAME}_split_b${BATCH}.onnx" --source "$OUT/preview_src" --out "$OUT/preview_onnx"

echo
echo "完成。目标平台: $PLATFORM"
echo "  ONNX : $OUT/${NAME}_split_b${BATCH}.onnx"
echo "  fp16 : $OUT/${NAME}_split_b${BATCH}_fp16.rknn"
echo "  int8 : $OUT/${NAME}_split_b${BATCH}_int8.rknn"
echo "  校准 : $OUT/${NAME}_split_b${BATCH}_int8.calib/  （源图目录: $CALIB_DIR）"
echo "板端: 拷贝 rknn_infer.py 与 .rknn 后执行:"
echo "  python3 rknn_infer.py --model ${NAME}_split_b${BATCH}_int8.rknn --source <images_dir> --out results"
