#!/bin/bash
# Phase 0: Validate scaling framework
# 5 architectures × 3 compute levels (T1=5s, T2=10s, T3=20s) = 15 runs
# SKIP_EVAL=1 skips torch.compile + int8 roundtrip (saves ~40s per run)
# Expected total: ~3-4 minutes
set -uo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0
mkdir -p results/phase0

CSV="results/phase0/results.csv"
echo "name,level,iters,final_train_loss,step_ms,params" > "$CSV"

run() {
    local name="$1"; shift
    local level="$1"; shift
    local iters="$1"; shift
    local warmup="$1"; shift

    echo "--- $name @ $level (${iters} steps) ---"
    mkdir -p "results/phase0/$name"
    # Use tee so output goes to both log file and stdout
    eval "export TRAIN_BATCH_TOKENS=65536 MAX_WALLCLOCK_SECONDS=0 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=10 ITERATIONS=$iters WARMUP_STEPS=$warmup SKIP_EVAL=1 $* RUN_ID=p0_$name"
    python3 train_gpt_golf.py 2>&1 | tee "results/phase0/$name/train.log"

    local log="results/phase0/$name/train.log"
    local train_loss=$(grep "^step:${iters}/" "$log" | grep -oP 'train_loss:\K[0-9.]+' || echo "N/A")
    if [ "$train_loss" = "N/A" ]; then
        train_loss=$(grep "val_loss:" "$log" | tail -1 | grep -oP 'val_loss:\K[0-9.]+' || echo "N/A")
    fi
    local step_ms=$(grep "step_avg" "$log" | tail -1 | grep -oP 'step_avg:\K[0-9.]+' || echo "N/A")
    local params=$(grep "model_params" "$log" | grep -oP 'model_params:\K[0-9]+' || echo "N/A")
    echo "$name,$level,$iters,$train_loss,$step_ms,$params" >> "$CSV"
    echo ">>> $name@$level: loss=$train_loss step=${step_ms}ms params=$params"
    echo ""
    sleep 1
}

echo ""
echo "============================================"
echo "PHASE 0: Scaling Law Validation"
echo "5 architectures × 3 compute levels = 15 runs"
echo "============================================"
echo ""

# T1: 5 seconds
echo "=== T1: 5-second runs (19 steps) ==="
run baseline_T1     T1 19 3 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2
run recur_small_T1  T1 19 3 NUM_UNIQUE_BLOCKS=3 NUM_EFFECTIVE_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2
run recur_deep_T1   T1 19 3 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2
run swiglu_deep_T1  T1 19 3 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1
run wide_T1         T1 19 3 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=4 MLP_MULT=2 USE_SWIGLU=1

# T2: 10 seconds
echo ""
echo "=== T2: 10-second runs (38 steps) ==="
run baseline_T2     T2 38 5 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2
run recur_small_T2  T2 38 5 NUM_UNIQUE_BLOCKS=3 NUM_EFFECTIVE_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2
run recur_deep_T2   T2 38 5 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2
run swiglu_deep_T2  T2 38 5 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1
run wide_T2         T2 38 5 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=4 MLP_MULT=2 USE_SWIGLU=1

# T3: 20 seconds
echo ""
echo "=== T3: 20-second runs (76 steps) ==="
run baseline_T3     T3 76 8 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2
run recur_small_T3  T3 76 8 NUM_UNIQUE_BLOCKS=3 NUM_EFFECTIVE_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2
run recur_deep_T3   T3 76 8 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2
run swiglu_deep_T3  T3 76 8 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1
run wide_T3         T3 76 8 NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=4 MLP_MULT=2 USE_SWIGLU=1

echo ""
echo "============================================"
echo "PHASE 0 COMPLETE"
echo "============================================"
echo ""
column -t -s',' "$CSV"
echo ""
echo "=== RANK STABILITY ==="
echo "T1: $(grep ',T1,' "$CSV" | sort -t',' -k4 -n | cut -d',' -f1 | sed 's/_T1//' | tr '\n' ' ')"
echo "T2: $(grep ',T2,' "$CSV" | sort -t',' -k4 -n | cut -d',' -f1 | sed 's/_T2//' | tr '\n' ' ')"
echo "T3: $(grep ',T3,' "$CSV" | sort -t',' -k4 -n | cut -d',' -f1 | sed 's/_T3//' | tr '\n' ' ')"
