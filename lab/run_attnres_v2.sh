#!/bin/bash
# AttnRes Phase 2 experiments: 500-step screening of all variants + 2000-step extensions
# All runs save checkpoints every 250 steps for resumability
# Validate every 50 steps to get fine-grained curves

set -e
cd /root/parameter-golf

COMMON="ITERATIONS=2000 VAL_LOSS_EVERY=50 TRAIN_LOG_EVERY=100 CHECKPOINT_EVERY=250 MAX_WALLCLOCK_SECONDS=0"

echo "========================================"
echo "Phase 2A: Baseline 2000 steps"
echo "========================================"
RUN_ID=p2_baseline_2000 $COMMON \
  python train_gpt.py

echo "========================================"
echo "Phase 2B: Value Residual (layer 0) 2000 steps"
echo "========================================"
RUN_ID=p2_vr_2000 ATTNRES_MODE=value_residual $COMMON \
  python train_gpt.py

echo "========================================"
echo "Phase 2C: Value Residual Gated 2000 steps"
echo "========================================"
RUN_ID=p2_vr_gated_2000 ATTNRES_MODE=value_residual_gated $COMMON \
  python train_gpt.py

echo "========================================"
echo "Phase 2D: Value Residual Mid (layer N/2) 2000 steps"
echo "========================================"
RUN_ID=p2_vr_mid_2000 ATTNRES_MODE=value_residual_mid $COMMON \
  python train_gpt.py

echo "========================================"
echo "Phase 2E: Weighted (scalar softmax) 2000 steps"
echo "========================================"
RUN_ID=p2_weighted_2000 ATTNRES_MODE=weighted $COMMON \
  python train_gpt.py

echo "========================================"
echo "Phase 2F: Weighted Vector (per-dim) 2000 steps"
echo "========================================"
RUN_ID=p2_weighted_vec_2000 ATTNRES_MODE=weighted_vector $COMMON \
  python train_gpt.py

echo "========================================"
echo "All Phase 2 experiments complete."
echo "Check logs/ for results."
echo "========================================"
