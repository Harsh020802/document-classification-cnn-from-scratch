#!/bin/bash
# Detached stage launcher, macOS-compatible (no setsid -- that is Linux-only).
# `nohup` plus disown detaches the job from the calling shell so it survives exit.
#   ./scripts/launch_stage.sh determinism "--epochs 3"
cd "$(dirname "$0")/.."
STAGE="$1"; EXTRA="${2:-}"
mkdir -p outputs/ablation_logs
LOG="outputs/ablation_logs/_stage_$STAGE.log"
EXTRA="$EXTRA" nohup ./scripts/16_run_ablation.sh "$STAGE" > "$LOG" 2>&1 &
PID=$!
disown "$PID" 2>/dev/null || true
echo "launched stage '$STAGE' (pid $PID) -> $LOG"
