#!/usr/bin/env bash
# metdist launcher — same contract as train_metdist.sh (start/status/tail/stop/foreground)
#   DETDIST_TAG=xxx ./train_detdist_v2.sh start --name xxx   # coexisting runs
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # .../ultralytics-depth
WS="$ROOT/hzh_workspace"
PY=/home/kang/miniforge3/envs/yolo/bin/python
ENTRY="$WS/tools/train/train_metdist.py"
LOGDIR="$WS/logs"
TAG="${DETDIST_TAG:-metdist}"
PIDFILE="$LOGDIR/$TAG.pid"
LATEST="$LOGDIR/${TAG}_latest.log"
mkdir -p "$LOGDIR"

export PYTHONPATH="$WS/libs:$ROOT"
export YOLO_CONFIG_DIR="$WS/env/cfg"
export MPLCONFIGDIR="$WS/env/cfg/mpl"
export TORCH_HOME="$WS/env/cfg/torch"
mkdir -p "$MPLCONFIGDIR" "$TORCH_HOME"
cd "$WS"

SUB=start
case "${1:-}" in
  start|status|tail|stop|foreground) SUB="$1"; shift ;;
  *) [ $# -gt 0 ] && SUB=start ;;
esac

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "$SUB" in
  start)
    if running; then
      echo "already running: pid=$(cat "$PIDFILE")  (use 'stop' first)"; exit 1
    fi
    TS="$(date +%Y%m%d-%H%M%S)"
    LOG="$LOGDIR/${TAG}_${TS}.log"
    setsid bash -c 'echo $$ > "$1"; shift; exec "$@"' _ "$PIDFILE" \
      "$PY" "$ENTRY" "$@" >"$LOG" 2>&1 &
    sleep 2
    ln -sfn "$LOG" "$LATEST"
    if running; then
      echo "started pid=$(cat "$PIDFILE")"
      echo "log    : $LOG"
      echo "latest : $LATEST"
    else
      echo "failed to start, last lines:"; tail -n 30 "$LOG"; exit 1
    fi
    ;;
  foreground)
    ln -sfn "$LOGDIR/${TAG}_fg.log" "$LATEST"
    exec "$PY" "$ENTRY" "$@" 2>&1 | tee -a "$LOGDIR/${TAG}_fg.log"
    ;;
  status)
    if running; then echo "running pid=$(cat "$PIDFILE")"; else echo "not running"; fi
    [ -f "$LATEST" ] && { echo "log: $(readlink -f "$LATEST")"; tail -n 5 "$(readlink -f "$LATEST")"; }
    ;;
  tail)
    exec tail -n 50 -f "$(readlink -f "$LATEST")"
    ;;
  stop)
    if running; then
      PID="$(cat "$PIDFILE")"
      kill "$PID" 2>/dev/null || true
      for _ in $(seq 1 20); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
      kill -9 "$PID" 2>/dev/null || true
      echo "stopped pid=$PID"
    else
      echo "not running"
    fi
    ;;
  *)
    echo "usage: $0 {start|foreground|status|tail|stop} [--key value ...]" >&2; exit 2 ;;
esac