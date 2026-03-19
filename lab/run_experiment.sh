#!/usr/bin/env bash
# Usage: lab/run_experiment.sh <experiment_name> <max_steps> [extra_env...]
#
# Runs train_gpt.py with standardized settings for 1-GPU development.
# Results go to results/<experiment_name>/train.log and logs/<experiment_name>.txt
#
# Examples:
#   lab/run_experiment.sh baseline_check 200
#   MATRIX_LR=0.08 lab/run_experiment.sh lr_test 500
#   NUM_LAYERS=12 MODEL_DIM=448 lab/run_experiment.sh arch_12x448 200

set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:?Usage: lab/run_experiment.sh <name> <max_steps>}"
MAX_STEPS="${2:?Usage: lab/run_experiment.sh <name> <max_steps>}"

# Convert steps to approximate wall-clock seconds on L40S (~3.33s/step)
# Add 15% buffer for validation + overhead
WALLCLOCK=$(python3 -c "import math; print(math.ceil($MAX_STEPS * 3.4 * 1.15))")

# Standardized settings for reproducible comparisons
export RUN_ID="${NAME}"
export ITERATIONS="${MAX_STEPS}"
export MAX_WALLCLOCK_SECONDS="${WALLCLOCK}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-50}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}"
export SEED="${SEED:-1337}"

# Print experiment config
echo "============================================"
echo "Experiment: ${NAME}"
echo "Max steps:  ${MAX_STEPS}"
echo "Wallclock:  ${WALLCLOCK}s (~$(python3 -c "print(f'{$WALLCLOCK/60:.1f}')")min)"
echo "============================================"
echo "Env overrides:"
for var in MATRIX_LR SCALAR_LR EMBED_LR NUM_LAYERS MODEL_DIM NUM_HEADS NUM_KV_HEADS \
           MLP_MULT WARMDOWN_ITERS WARMUP_STEPS LOGIT_SOFTCAP QK_GAIN_INIT ROPE_BASE \
           MUON_MOMENTUM MUON_BACKEND_STEPS GRAD_CLIP_NORM TIED_EMBED_LR TIED_EMBED_INIT_STD; do
    if [ -n "${!var+x}" ]; then
        echo "  ${var}=${!var}"
    fi
done
echo "============================================"

mkdir -p results/"${NAME}" logs

# Run training
python3 train_gpt.py 2>&1 | tee logs/"${NAME}.txt"

# Copy the log into results dir for self-contained experiment records
if [ -f "logs/${NAME}.txt" ]; then
    cp "logs/${NAME}.txt" "results/${NAME}/train.log"
fi

# Extract final metrics
echo ""
echo "=== Results ==="
if grep -q "val_bpb" "logs/${NAME}.txt"; then
    echo "Final val_bpb:"
    grep "val_bpb" "logs/${NAME}.txt" | tail -1
fi
if grep -q "final_int8_zlib_roundtrip" "logs/${NAME}.txt"; then
    echo "Post-quant val_bpb:"
    grep "final_int8_zlib_roundtrip_exact" "logs/${NAME}.txt" | tail -1
fi
echo "==============="
