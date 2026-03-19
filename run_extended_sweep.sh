#!/bin/bash
# Extended sweep: ~7 hours of experiments for Parameter Golf
# Runs after run_quick_sweep.sh completes
set -uo pipefail  # no -e: individual experiment failures must not kill the sweep
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0
BASE="TRAIN_BATCH_TOKENS=524288 MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=100"

run() {
    local name="$1"; shift
    local iters="${1}"; shift
    local val_every=$((iters / 4))
    [ "$val_every" -lt 50 ] && val_every=50
    echo "============================================"
    echo "EXP: $name (${iters} steps, val every ${val_every})"
    echo "============================================"
    mkdir -p "results/$name"
    eval "export $BASE VAL_LOSS_EVERY=$val_every ITERATIONS=$iters $* RUN_ID=$name"
    if python3 train_gpt_golf.py 2>&1 | tee "results/$name/train.log"; then
        bpb=$(grep "final_int8_zlib_roundtrip_exact" "results/$name/train.log" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
        echo ">>> $name: int8_bpb=$bpb"
        echo "$name,$iters,$bpb" >> results/sweep_results.csv
    else
        echo ">>> $name: CRASHED (exit $?)"
        echo "$name,$iters,CRASHED" >> results/sweep_results.csv
    fi
    # Brief cooldown to release GPU memory
    sleep 2
    echo ""
}

echo "name,iters,int8_bpb" > results/sweep_results.csv

# ============================================================
# PHASE A: Architecture search - recurrence ratios (~80 min)
# ============================================================

# Vary unique blocks and effective layers at different dims
run "a01_2x8_512"   500 NUM_UNIQUE_BLOCKS=2 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2
run "a02_2x12_512"  500 NUM_UNIQUE_BLOCKS=2 NUM_EFFECTIVE_LAYERS=24 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2
run "a03_5x3_576"   500 NUM_UNIQUE_BLOCKS=5 NUM_EFFECTIVE_LAYERS=15 MODEL_DIM=576 NUM_HEADS=9 NUM_KV_HEADS=3 MLP_MULT=2
run "a04_6x3_512"   500 NUM_UNIQUE_BLOCKS=6 NUM_EFFECTIVE_LAYERS=18 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2
run "a05_4x5_608"   500 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=20 MODEL_DIM=608 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2
run "a06_3x8_672"   500 NUM_UNIQUE_BLOCKS=3 NUM_EFFECTIVE_LAYERS=24 MODEL_DIM=672 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2
run "a07_4x6_640"   500 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=24 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2
run "a08_5x4_576"   500 NUM_UNIQUE_BLOCKS=5 NUM_EFFECTIVE_LAYERS=20 MODEL_DIM=576 NUM_HEADS=9 NUM_KV_HEADS=3 MLP_MULT=2
run "a09_2x10_640"  500 NUM_UNIQUE_BLOCKS=2 NUM_EFFECTIVE_LAYERS=20 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2
run "a10_3x6_640"   500 NUM_UNIQUE_BLOCKS=3 NUM_EFFECTIVE_LAYERS=18 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2

# ============================================================
# PHASE B: MLP multiplier variations (~40 min)
# ============================================================

run "b01_4x4_640_mlp3" 500 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=3
run "b02_4x4_512_mlp3" 500 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3
run "b03_4x4_576_mlp3" 500 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=576 NUM_HEADS=9 NUM_KV_HEADS=3 MLP_MULT=3
run "b04_3x6_512_mlp3" 500 NUM_UNIQUE_BLOCKS=3 NUM_EFFECTIVE_LAYERS=18 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3
run "b05_4x4_640_mlp4" 500 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=4

# ============================================================
# PHASE C: SwiGLU combinations (~40 min)
# ============================================================

run "c01_swiglu_3x6_704"  500 NUM_UNIQUE_BLOCKS=3 NUM_EFFECTIVE_LAYERS=18 MODEL_DIM=704 NUM_HEADS=11 NUM_KV_HEADS=1 MLP_MULT=2 USE_SWIGLU=1
run "c02_swiglu_4x6_640"  500 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=24 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1
run "c03_swiglu_3x8_672"  500 NUM_UNIQUE_BLOCKS=3 NUM_EFFECTIVE_LAYERS=24 MODEL_DIM=672 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 USE_SWIGLU=1
run "c04_swiglu_2x12_512" 500 NUM_UNIQUE_BLOCKS=2 NUM_EFFECTIVE_LAYERS=24 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 USE_SWIGLU=1
run "c05_swiglu_5x4_576"  500 NUM_UNIQUE_BLOCKS=5 NUM_EFFECTIVE_LAYERS=20 MODEL_DIM=576 NUM_HEADS=9 NUM_KV_HEADS=3 MLP_MULT=2 USE_SWIGLU=1

# ============================================================
# PHASE D: LR scaling experiments (~40 min)
# Best architecture from q-sweep (using 4x4@640 as proxy)
# ============================================================

run "d01_lr0.5x"  500 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1 TIED_EMBED_LR=0.025 MATRIX_LR=0.02 SCALAR_LR=0.02
run "d02_lr1.5x"  500 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1 TIED_EMBED_LR=0.075 MATRIX_LR=0.06 SCALAR_LR=0.06
run "d03_lr2x"    500 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1 TIED_EMBED_LR=0.10 MATRIX_LR=0.08 SCALAR_LR=0.08
run "d04_cosine_lr1.5x" 500 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1 LR_SCHEDULE=cosine TIED_EMBED_LR=0.075 MATRIX_LR=0.06 SCALAR_LR=0.06
run "d05_warmdown20" 500 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1 WARMDOWN_FRAC=0.2

# ============================================================
# PHASE E: QAT schedule variations (~32 min)
# ============================================================

run "e01_qat30"   500 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1 QAT_START_FRAC=0.3
run "e02_qat70"   500 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1 QAT_START_FRAC=0.7
run "e03_qat20"   500 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1 QAT_START_FRAC=0.2
run "e04_qat_cosine" 500 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1 QAT_START_FRAC=0.5 LR_SCHEDULE=cosine

# ============================================================
# PHASE F: Medium runs (1000 steps) on top configs (~60 min)
# ============================================================

run "f01_med_4x4_640_swiglu"     1000 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1
run "f02_med_3x6_704_swiglu"     1000 NUM_UNIQUE_BLOCKS=3 NUM_EFFECTIVE_LAYERS=18 MODEL_DIM=704 NUM_HEADS=11 NUM_KV_HEADS=1 MLP_MULT=2 USE_SWIGLU=1
run "f03_med_4x6_640_swiglu"     1000 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=24 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1
run "f04_med_combined_best"      1000 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1 QAT_START_FRAC=0.5 LR_SCHEDULE=cosine

# ============================================================
# PHASE G: Full runs (2000 steps) on winners (~60 min)
# ============================================================

run "g01_full_combined"   2000 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1 QAT_START_FRAC=0.5 LR_SCHEDULE=cosine
run "g02_full_deep"       2000 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=24 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1 QAT_START_FRAC=0.5 LR_SCHEDULE=cosine
run "g03_full_wide"       2000 NUM_UNIQUE_BLOCKS=3 NUM_EFFECTIVE_LAYERS=18 MODEL_DIM=704 NUM_HEADS=11 NUM_KV_HEADS=1 MLP_MULT=2 USE_SWIGLU=1 QAT_START_FRAC=0.5 LR_SCHEDULE=cosine

# ============================================================
# PHASE H: Extra long runs (4000 steps) to see ceiling (~60 min)
# ============================================================

run "h01_4k_combined"  4000 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1 QAT_START_FRAC=0.5 LR_SCHEDULE=cosine
run "h02_4k_deep"      4000 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=24 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1 QAT_START_FRAC=0.5 LR_SCHEDULE=cosine

echo ""
echo "============================================"
echo "EXTENDED SWEEP COMPLETE"
echo "============================================"
echo ""
echo "ALL RESULTS:"
sort -t',' -k3 -n results/sweep_results.csv | column -t -s','
