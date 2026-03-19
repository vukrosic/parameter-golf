#!/bin/bash
set -euo pipefail

cd /root/parameter-golf

COMMON="ITERATIONS=400 VAL_LOSS_EVERY=50 TRAIN_LOG_EVERY=100 MAX_WALLCLOCK_SECONDS=0 MATRIX_LR=0.06"

run_one() {
  local run_id="$1"
  shift
  echo "========================================"
  echo "Running ${run_id}"
  echo "========================================"
  env RUN_ID="${run_id}" ${COMMON} "$@" python train_gpt.py
}

run_one heads84_lr06_400 NUM_HEADS=8 NUM_KV_HEADS=4
run_one heads88_lr06_400 NUM_HEADS=8 NUM_KV_HEADS=8
run_one heads168_lr06_400 NUM_HEADS=16 NUM_KV_HEADS=8
run_one heads82_lr06_400 NUM_HEADS=8 NUM_KV_HEADS=2
