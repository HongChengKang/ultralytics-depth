#!/usr/bin/env bash
# COCO_PF YOLO26-DetDepth：检测从 yaml 起、深度头从 teacher 迁移；单卡 A6000（batch=80，约 9 GB 显存）。
#
#   ./train_coco_pf.sh                  # 前台跑（400 epochs，patience=20）
#   ./train_coco_pf.sh epochs=2         # 任意 ultralytics 覆盖参数，如 key=value ...
#   ./train_coco_pf.sh start [k=v ...]  # 后台跑（setsid）：关掉 SSH 也不停
#   ./train_coco_pf.sh status           # 是否存活 / 日志 / GPU / results.csv 最后一行
#   ./train_coco_pf.sh tail             # 跟踪后台日志
#   ./train_coco_pf.sh stop             # 停掉整个进程组
#
# 后台记账：hzh_workspace/logs/pf_train.pid、pf_train_latest.log、pf_train_<timestamp>.log
# 默认结果目录：hzh_workspace/runs/detdepth_coco_pf（v2 正式；v1 旧结果在 ..._old）
# status 会自动取最新的 detdepth_coco_pf 或 -N（排除 _old）
# 可用 RUNDIR=... 覆盖 status 查看的目录，例如查看旧 v1：
#   RUNDIR=$P/runs/detdepth_coco_pf_old ./train_coco_pf.sh status
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P="$ROOT/hzh_workspace"
PY=/home/kang/miniforge3/envs/yolo/bin/python
ENTRY="$P/tools/train/train_detdepth_pf.py"
LOGDIR="$P/logs"
PIDFILE="$LOGDIR/pf_train.pid"
LATEST="$LOGDIR/pf_train_latest.log"
# status 默认看最新的 detdepth_coco_pf 或 -N（排除 _old）；显式 RUNDIR=... 优先。
if [ -z "${RUNDIR:-}" ]; then
  RUNDIR="$P/runs/detdepth_coco_pf"
  if [ -d "$P/runs" ]; then
    newest="$(find "$P/runs" -maxdepth 1 -type d \( -name 'detdepth_coco_pf' -o -name 'detdepth_coco_pf-[0-9]*' \) -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
    [ -n "$newest" ] && RUNDIR="$newest"
  fi
fi
SELF="$(basename "$0")"

export PYTHONPATH="$P/env/pylibs:$ROOT/hck_workspace/detdepth:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_CONFIG_DIR="$P/env/cfg"
export MPLCONFIGDIR="$P/env/cfg/mpl"
export TORCH_HOME="$P/env/cfg/torch"
mkdir -p "$LOGDIR" "$P/env/cfg/mpl" "$P/env/cfg/torch"

pid_of()  { cat "$PIDFILE" 2>/dev/null || true; }
alive()   { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }

case "${1:-}" in
  start)
    shift
    old="$(pid_of)"
    if alive "$old"; then
      echo "already running: pid $old  (./$SELF status | ./$SELF stop)" >&2; exit 1
    fi
    LOG="$LOGDIR/pf_train_$(date +%Y%m%d_%H%M%S).log"
    setsid bash -c 'echo $$ > "$1"; shift; exec "$@"' _ "$PIDFILE" \
      "$PY" -u "$ENTRY" "$@" >> "$LOG" 2>&1 < /dev/null &
    ln -sfn "$LOG" "$LATEST"
    sleep 3
    pid="$(pid_of)"
    if alive "$pid"; then
      echo "started: pid $pid"
      echo "log:     $LOG"
      echo "latest:  $LATEST"
      echo "now you can close this SSH window; check later with:  ./$SELF status"
    else
      echo "FAILED to stay alive, last 40 log lines:" >&2
      tail -n 40 "$LOG" >&2 || true
      rm -f "$PIDFILE"
      exit 1
    fi
    ;;

  status)
    pid="$(pid_of)"
    if alive "$pid"; then
      echo "RUNNING  pid=$pid  uptime=$(ps -o etime= -p "$pid" | tr -d ' ')"
    else
      echo "NOT RUNNING"
    fi
    if alive "$pid"; then
      echo "log:      $(readlink -f "$LATEST" 2>/dev/null || echo '(none)')"
    else
      echo "log:      (not running; latest artifact: $(readlink -f "$LATEST" 2>/dev/null || echo 'none'))"
    fi
    echo "run dir:  $RUNDIR"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader | sed 's/^/gpu:      /'
    if [ -f "$RUNDIR/results.csv" ]; then
      echo "--- results.csv (header + last row) ---"
      sed -n '1p' "$RUNDIR/results.csv"
      tail -n 1 "$RUNDIR/results.csv"
      echo "epochs done: $(( $(wc -l < "$RUNDIR/results.csv") - 1 ))"
    else
      echo "--- results.csv not written yet ---"
    fi
    if alive "$pid" && [ -f "$LATEST" ]; then
      echo "--- last log line ---"
      tr '\r' '\n' < "$LATEST" | grep -v '^\s*$' | tail -n 3
    fi
    ;;

  tail)
    [ -f "$LATEST" ] || { echo "no log at $LATEST" >&2; exit 1; }
    tail -f "$LATEST" | tr -d '\r'
    ;;

  stop)
    pid="$(pid_of)"
    if ! alive "$pid"; then echo "not running"; rm -f "$PIDFILE"; exit 0; fi
    pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
    echo "stopping pid=$pid pgid=$pgid (+ every DDP child in that group)"
    kill -TERM -- "-$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do alive "$pid" || break; sleep 1; done
    if alive "$pid"; then
      echo "still alive after 20s, SIGKILL"
      kill -KILL -- "-$pgid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
    echo "stopped"
    ;;

  -h|--help|help)
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
    ;;

  *)  # no args or key=value ... -> foreground run
    exec "$PY" -u "$ENTRY" "$@"
    ;;
esac
