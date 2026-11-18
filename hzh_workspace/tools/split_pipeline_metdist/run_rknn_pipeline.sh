#!/usr/bin/env bash
set -euo pipefail

REPO=/home/kang/workspace/projects/yolo-depth/ultralytics-depth
WS=$REPO/hzh_workspace
YOLO_PY=/home/kang/miniforge3/envs/yolo/bin/python
RKNN_PY=/home/eav-nanjin/miniforge3/envs/rknn/bin/python
D="$(cd "$(dirname "$0")" && pwd)"

[ -f "$D/pipeline.env" ] && . "$D/pipeline.env"

PT=${PT:-runs/dist_pf_all/weights/best.pt}; PT=${PT/#\~/$HOME}
OUT=${OUT:-models/split_pf_11_11_3576}
PLATFORM=${PLATFORM:-rk3576}
BATCH=${BATCH:-1}
CALIB_DIR=${CALIB_DIR:-/mnt/data2/hzh/PF_11.9/images/train}
VAL_DIR=${VAL_DIR:-/mnt/data2/hzh/PF_11.9/images/val}
CALIB_NUM=${CALIB_NUM:-300}
CHECK_NUM=${CHECK_NUM:-40}

while [ $# -gt 0 ]; do
  case "$1" in
    --platform) PLATFORM=$2; shift 2;;
    --batch)    BATCH=$2; shift 2;;
    --imgsz)    IMGSZ=$2; shift 2;;
    --out)      OUT=$2; shift 2;;
    --pt)       PT=$2; shift 2;;
    --print)    PRINT=1; shift;;
    -*)         echo "unknown: $1" >&2; exit 1;;
    *)          PT=$1; shift;;
  esac
done

[[ "$PT" = /* ]] || PT=$WS/$PT
[[ "$OUT" = /* ]] || OUT=$WS/$OUT
[[ "$PLATFORM" == rk* ]] || { echo "PLATFORM must be rk3588/rk3576, got: $PLATFORM" >&2; exit 1; }
[ -f "$PT" ] || { echo "no checkpoint: $PT" >&2; exit 1; }

RUN_DIR=$(dirname "$(dirname "$PT")")
[ "$(basename "$(dirname "$PT")")" = "weights" ] || RUN_DIR=$(dirname "$PT")
NAME=$(basename "$RUN_DIR")

if [ "${IMGSZ:-auto}" = "auto" ]; then
  IMGSZ=$(awk '/^imgsz:/{sub(/^imgsz:[[:space:]]*/,""); gsub(/[][ ]/,""); gsub(/,/,"x"); print}' "$RUN_DIR/args.yaml" 2>/dev/null || true)
  IMGSZ=${IMGSZ:-640}
fi

SUFFIX=b${BATCH}_${PLATFORM#rk}
ONNX="$OUT/${NAME}_split_b${BATCH}.onnx"
FP16="$OUT/${NAME}_split_${SUFFIX}_fp16.rknn"
INT8="$OUT/${NAME}_split_${SUFFIX}_int8.rknn"
CALIB_WORK="$OUT/${NAME}_split_${SUFFIX}_int8.calib"

if [ "${PRINT:-0}" = 1 ]; then
  printf 'pt      %s\nout     %s\nname    %s\nplatform %s\nbatch   %s\nimgsz   %s\ncalib   %s (%s imgs)\nval     %s (%s imgs)\n' \
    "$PT" "$OUT" "$NAME" "$PLATFORM" "$BATCH" "$IMGSZ" "$CALIB_DIR" "$CALIB_NUM" "$VAL_DIR" "$CHECK_NUM"
  exit 0
fi

mkdir -p "$OUT"
echo "== 1/4 onnx  ($NAME, batch=$BATCH, imgsz=$IMGSZ)"
"$YOLO_PY" "$D/export_onnx_split.py" "$PT" "$ONNX" --batch "$BATCH" --imgsz "$IMGSZ"

echo "== 2/4 fp16  ($PLATFORM)"
"$RKNN_PY" "$D/convert_rknn.py" "$ONNX" "$FP16" --platform "$PLATFORM" --fp --check "$CHECK_NUM" --val "$VAL_DIR"

echo "== 3/4 int8  ($PLATFORM, $CALIB_NUM calib imgs)"
"$RKNN_PY" "$D/convert_rknn.py" "$ONNX" "$INT8" --platform "$PLATFORM" \
    --dataset "$CALIB_DIR" --num "$CALIB_NUM" --check "$CHECK_NUM" --val "$VAL_DIR"

echo "== 4/4 preview"
mkdir -p "$OUT/preview_src"
find "$VAL_DIR" -maxdepth 1 -type f | sort | awk 'NR % 100 == 1' | head -8 | xargs -I{} ln -sf {} "$OUT/preview_src/"
"$RKNN_PY" "$D/rknn_infer.py" --model "$ONNX" --source "$OUT/preview_src" --out "$OUT/preview_onnx"

echo "done: $PLATFORM batch=$BATCH imgsz=$IMGSZ"
ls -lh "$ONNX" "$FP16" "$INT8"
