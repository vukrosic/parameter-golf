#!/bin/bash
# Parameter Golf experiment runner for L40S
# Usage: bash run_experiments.sh [experiment_number]

set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0

# L40S config: single GPU, full batch size (matches 8xH100 token budget per step)
# grad_accum_steps = 8 // world_size = 8 (single GPU)
# micro_batch = 524288/8 = 65536 tokens = 64 seqs @1024 — fits easily in 48GB
BASE_ENV="TRAIN_BATCH_TOKENS=524288 MAX_WALLCLOCK_SECONDS=0 VAL_LOSS_EVERY=500 TRAIN_LOG_EVERY=100"

run_exp() {
    local name="$1"
    local iters="$2"
    local extra_env="$3"

    echo "============================================"
    echo "EXPERIMENT: $name"
    echo "ITERATIONS: $iters"
    echo "CONFIG: $extra_env"
    echo "============================================"

    mkdir -p "results/$name"

    eval "export $BASE_ENV $extra_env ITERATIONS=$iters RUN_ID=$name"
    python3 train_gpt_golf.py 2>&1 | tee "results/$name/train.log"

    # Copy artifacts
    cp -f final_model.int8.ptz "results/$name/" 2>/dev/null || true
    cp -f final_model.pt "results/$name/" 2>/dev/null || true

    echo ""
    echo "=== RESULTS for $name ==="
    grep -E "(val_bpb|final_int8)" "results/$name/train.log" | tail -5
    echo ""
}

case "${1:-all}" in
    0|baseline)
        # Phase 0: L40S baseline (original architecture, 2000 steps)
        run_exp "exp0_baseline" 2000 \
            "NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2"
        ;;

    1|sanity)
        # Phase 1A: Depth recurrence sanity check (3 unique -> 9 effective, same as baseline)
        run_exp "exp1_recurrence_sanity" 2000 \
            "NUM_UNIQUE_BLOCKS=3 NUM_EFFECTIVE_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2"
        ;;

    2|recurrence_d)
        # Phase 1D: 4 unique -> 16 effective @ dim=640
        run_exp "exp2_recurrence_d640" 2000 \
            "NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2"
        ;;

    3|recurrence_e)
        # Phase 1E: 3 unique -> 18 effective @ dim=704
        run_exp "exp3_recurrence_e704" 2000 \
            "NUM_UNIQUE_BLOCKS=3 NUM_EFFECTIVE_LAYERS=18 MODEL_DIM=704 NUM_HEADS=11 NUM_KV_HEADS=1 MLP_MULT=2"
        ;;

    4|swiglu)
        # Phase 3: SwiGLU on best recurrence config (placeholder: uses config D)
        run_exp "exp4_swiglu_d640" 2000 \
            "NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1"
        ;;

    5|qat)
        # Phase 2: QAT on best config (50% start)
        run_exp "exp5_qat_d640" 2000 \
            "NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 QAT_START_FRAC=0.5"
        ;;

    6|combined)
        # Phase 5: Combined: recurrence + SwiGLU + QAT
        run_exp "exp6_combined" 2000 \
            "NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1 QAT_START_FRAC=0.5"
        ;;

    7|cosine)
        # Phase 4: Cosine LR schedule on combined
        run_exp "exp7_cosine_combined" 2000 \
            "NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1 QAT_START_FRAC=0.5 LR_SCHEDULE=cosine"
        ;;

    8|long)
        # Phase 5: Long validation run (more iterations, ~1-2 hours)
        run_exp "exp8_long_validation" 8000 \
            "NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1 QAT_START_FRAC=0.5 LR_SCHEDULE=cosine VAL_LOSS_EVERY=1000"
        ;;

    all)
        for i in 0 1 2 3 4 5 6 7; do
            bash "$0" "$i"
        done
        ;;

    *)
        echo "Usage: $0 {0|baseline|1|sanity|2|recurrence_d|3|recurrence_e|4|swiglu|5|qat|6|combined|7|cosine|8|long|all}"
        exit 1
        ;;
esac
