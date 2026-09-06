#!/bin/bash
# Sequential ablation run sequence (reply-15 Part 3). NOT parallel: parallel runs on
# 8 GB with MPS contend and corrupt the timings.
set -e
cd "$(dirname "$0")/.."
mkdir -p outputs/ablation_logs

run () {   # run <head> <seed> <tag>
  echo "=== $1 seed $2 ($3) === $(date '+%H:%M:%S')"
  ./run.sh scripts/09_train.py --head "$1" --seed "$2" --tag "$3" ${EXTRA:-} \
    > "outputs/ablation_logs/$1_seed$2_$3.log" 2>&1
}

case "$1" in
  determinism)
    run gap3 0 det1
    run gap3 0 det2
    ;;
  floor)
    run gap3 0 floor0
    run gap3 1 floor1
    run gap3 2 floor2
    ;;
  arms)
    # gap3 seed 0 is reused from the floor stage -- not retrained.
    run gap1        0 abl
    run gap1_hidden 0 abl
    run gap7        0 abl
    ;;
esac
echo "STAGE $1 COMPLETE $(date '+%H:%M:%S')"
