#!/usr/bin/env bash
# 伪深度生成入口：给 YOLO 检测数据集补 <root>/depth/<split>/<stem>.png（uint16，毫米）。
#
#   ./gen_depth.sh --root /mnt/data2/COCO_PF --splits train val
#   ./gen_depth.sh --root /mnt/data2/COCO_PF --splits val --batch 16 --device 0
#   ./gen_depth.sh --root /mnt/data2/COCO_PF --splits train --weights /path/to/teacher.pt
#
# 说明：
#   - 已存在的 depth PNG 会被自动跳过，所以重复跑只会补新增图片；
#   - teacher 默认用仓库根的 model_common/yolo26n-depth.pt，不依赖脚本内部的默认路径；
#   - --root 必须显式给，避免误写到默认数据集。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P="$ROOT/hzh_workspace"
PY=/home/kang/miniforge3/envs/yolo/bin/python
TEACHER="$ROOT/model_common/yolo26n-depth.pt"

export PYTHONPATH="$P/env/pylibs:$ROOT/hck_workspace/detdepth:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_CONFIG_DIR="$P/env/cfg"
export MPLCONFIGDIR="$P/env/cfg/mpl"
export TORCH_HOME="$P/env/cfg/torch"
mkdir -p "$P/env/cfg/mpl" "$P/env/cfg/torch"

usage() {
  sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
}

if [ "$#" -eq 0 ]; then
  usage
  echo
  echo "ERROR: 至少给 --root（例如 --root /mnt/data2/COCO_PF）" >&2
  exit 1
fi

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

has_root=0
has_weights=0
for arg in "$@"; do
  case "$arg" in
    --root|--root=*)       has_root=1 ;;
    --weights|--weights=*) has_weights=1 ;;
  esac
done

if [ "$has_root" -eq 0 ]; then
  echo "ERROR: --root 必须显式指定，例如 --root /mnt/data2/COCO_PF" >&2
  exit 2
fi

args=("$@")
if [ "$has_weights" -eq 0 ]; then
  if [ ! -f "$TEACHER" ]; then
    echo "ERROR: teacher 不存在：$TEACHER" >&2
    echo "       可显式传 --weights /path/to/teacher.pt" >&2
    exit 3
  fi
  args+=(--weights "$TEACHER")
fi

exec "$PY" -u "$P/tools/data/gen_pseudo_depth.py" "${args[@]}"
