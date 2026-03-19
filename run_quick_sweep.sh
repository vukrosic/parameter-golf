#!/bin/bash
# Quick 500-step sweep for architecture search (~8 min each)
set -uo pipefail  # no -e: crashes must not kill the sweep
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0
BASE="TRAIN_BATCH_TOKENS=524288 MAX_WALLCLOCK_SECONDS=0 VAL_LOSS_EVERY=250 TRAIN_LOG_EVERY=100 ITERATIONS=500"

run() {
    local name="$1"; shift
    echo "============================================"
    echo "QUICK EXPERIMENT: $name"
    echo "============================================"
    mkdir -p "results/$name"
    eval "export $BASE $* RUN_ID=$name"
    if python3 train_gpt_golf.py 2>&1 | tee "results/$name/train.log"; then
        echo ""
        echo "=== RESULT $name ==="
        grep -E "(val_bpb|final_int8)" "results/$name/train.log" | tail -5
    else
        echo ">>> $name: CRASHED (exit $?)"
    fi
    sleep 2
    echo ""
}

# 1. Recurrence sanity: 3 unique -> 9 effective @512 (same depth, fewer params)
run "q1_recur_3x3_512" NUM_UNIQUE_BLOCKS=3 NUM_EFFECTIVE_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2

# 2. Recurrence D: 4 unique -> 16 effective @640 (wider + deeper)
run "q2_recur_4x4_640" NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2

# 3. Recurrence E: 3 unique -> 18 effective @704 (very wide, very deep)
run "q3_recur_3x6_704" NUM_UNIQUE_BLOCKS=3 NUM_EFFECTIVE_LAYERS=18 MODEL_DIM=704 NUM_HEADS=11 NUM_KV_HEADS=1 MLP_MULT=2

# 4. SwiGLU on config D
run "q4_swiglu_640" NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1

# 5. QAT on config D
run "q5_qat_640" NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 QAT_START_FRAC=0.5

# 6. Combined: recurrence D + SwiGLU + QAT
run "q6_combined" NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1 QAT_START_FRAC=0.5

# 7. Combined + cosine LR
run "q7_cosine" NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1 QAT_START_FRAC=0.5 LR_SCHEDULE=cosine

# 8. Baseline (no recurrence) for 500-step comparison
run "q0_baseline_500" NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2

echo ""
echo "============================================"
echo "ALL QUICK EXPERIMENTS COMPLETE"
echo "============================================"
for d in results/q*; do
    name=$(basename "$d")
    bpb=$(grep "final_int8_zlib_roundtrip_exact" "$d/train.log" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    echo "$name: int8_bpb=$bpb"
done
