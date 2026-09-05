#!/bin/bash
# Live view of the newest training run. Usage: ./watch_run.sh
RUN=$(ls -td outputs/runs/*/ 2>/dev/null | head -1)
[ -z "$RUN" ] && { echo "no runs found"; exit 1; }
while true; do
  clear
  echo "run: $RUN"
  echo "alive: $(pgrep -f '09_train.py' | wc -l | tr -d ' ') process(es)   $(date '+%H:%M:%S')"
  echo
  printf "%3s %10s %9s %8s %9s %8s %8s\n" ep "tr loss" "tr acc" "tr F1" "va loss" "va acc" "va F1"
  echo "----------------------------------------------------------------------"
  awk -F, 'NR>1{printf "%3d %10.4f %9.4f %8.4f %9.4f %8.4f %8.4f\n",$1,$3,$4,$5,$6,$7,$8}' "$RUN/metrics.csv"
  echo
  BEST=$(awk -F, 'NR>1{if($8>m){m=$8;e=$1}}END{printf "%.4f @ epoch %d",m,e}' "$RUN/metrics.csv")
  echo "best val macro-F1: $BEST     (logreg baseline: 0.5599)"
  pgrep -f '09_train.py' >/dev/null || { echo; echo "RUN FINISHED"; break; }
  sleep 10
done
