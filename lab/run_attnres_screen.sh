#!/bin/bash
# Sequential 500-step screening of all AttnRes variants
# ONE experiment at a time to avoid GPU contention
set -e
cd /root/parameter-golf

COMMON="ITERATIONS=500 VAL_LOSS_EVERY=50 TRAIN_LOG_EVERY=100 CHECKPOINT_EVERY=500 MAX_WALLCLOCK_SECONDS=0"

run_one() {
    local name=$1
    local mode=$2
    echo ""
    echo "========================================"
    echo "[$name] ATTNRES_MODE=$mode  (500 steps)"
    echo "========================================"
    RUN_ID=$name ATTNRES_MODE=$mode $COMMON python train_gpt.py 2>&1 | \
        grep -E "^(step:.*(val_bpb|train_loss)|final_int8_zlib_roundtrip_exact)"
    echo "[$name] DONE"
}

# baseline_500 and attnres_vr_500b already exist from earlier runs
echo "Skipping baseline_500 (already have: 1.4540 BPB int8)"
echo "Skipping value_residual (already have: 1.4582 BPB int8)"

run_one p2_vr_gated_500    value_residual_gated
run_one p2_vr_mid_500       value_residual_mid
run_one p2_weighted_500     weighted
run_one p2_wvec_500         weighted_vector

echo ""
echo "========================================"
echo "ALL SCREENING RUNS COMPLETE"
echo "Compare results in logs/p2_*.txt"
echo "========================================"
