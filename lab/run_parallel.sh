#!/usr/bin/env bash
# Runs experiments from queue in parallel across all available GPUs.
# Each GPU gets experiments in round-robin order from the queue.
# Skips already-completed experiments (those with results/<name>/train.log).
set -uo pipefail
cd "$(dirname "$0")/.."

QUEUE_FILE="${1:-lab/queue_ordered.txt}"
NUM_GPUS="${2:-8}"
FAIL_LOG="lab/failed_experiments.log"

echo "===== Parallel queue runner started at $(date) ====="
echo "GPUs: $NUM_GPUS"
echo "Queue: $QUEUE_FILE"

# Collect all pending experiments
pending=()
while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// /}" ]] && continue

    name=$(echo "$line" | awk '{print $1}')

    # Skip if already done (check for summary.json or train.log)
    if [ -d "results/$name" ] && { [ -f "results/$name/train.log" ] || [ -f "results/$name/summary.json" ]; }; then
        echo "SKIP (done): $name"
        continue
    fi

    pending+=("$line")
done < "$QUEUE_FILE"

echo "Pending experiments: ${#pending[@]}"
echo ""

# Function to run one experiment on a specific GPU
run_on_gpu() {
    local gpu_id=$1
    local line="$2"
    local name=$(echo "$line" | awk '{print $1}')
    local steps=$(echo "$line" | awk '{print $2}')
    local envvars=$(echo "$line" | awk '{for(i=3;i<=NF;i++) printf "%s ", $i}')

    echo "[GPU $gpu_id] Starting: $name ($steps steps) [$envvars]"

    local env_cmd="export CUDA_VISIBLE_DEVICES=$gpu_id;"
    for ev in $envvars; do
        [[ "$ev" =~ ^# ]] && break
        env_cmd="$env_cmd export $ev;"
    done

    local start_time=$(date +%s)
    (
        eval "$env_cmd"
        bash lab/run_experiment.sh "$name" "$steps"
    ) > "logs/${name}_gpu${gpu_id}.txt" 2>&1
    local exit_code=$?
    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))

    if [ $exit_code -ne 0 ]; then
        echo "[GPU $gpu_id] FAILED: $name (exit=$exit_code, ${elapsed}s)"
        echo "$(date) | $name | exit=$exit_code | ${elapsed}s | GPU=$gpu_id | $envvars" >> "$FAIL_LOG"
    else
        echo "[GPU $gpu_id] DONE: $name (${elapsed}s)"
    fi
}

mkdir -p logs

# Launch experiments in waves of NUM_GPUS
idx=0
total=${#pending[@]}

while [ $idx -lt $total ]; do
    echo ""
    echo "===== Wave starting at $(date '+%H:%M:%S') (experiments $idx-$((idx + NUM_GPUS - 1)) of $total) ====="

    pids=()
    gpu=0
    batch_end=$((idx + NUM_GPUS))
    if [ $batch_end -gt $total ]; then
        batch_end=$total
    fi

    for (( i=idx; i<batch_end; i++ )); do
        run_on_gpu $gpu "${pending[$i]}" &
        pids+=($!)
        gpu=$((gpu + 1))
    done

    # Wait for all in this wave
    for pid in "${pids[@]}"; do
        wait $pid
    done

    idx=$batch_end
    echo "===== Wave complete at $(date '+%H:%M:%S') ====="
done

echo ""
echo "===== Parallel queue runner finished at $(date) ====="
echo "Total experiments: $total"
