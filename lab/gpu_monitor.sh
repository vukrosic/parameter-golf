#!/usr/bin/env bash
# GPU monitor: checks every 60s, reports status, restarts crashed jobs from queue.
# Usage: lab/gpu_monitor.sh [queue_file]
# Runs forever. Kill with Ctrl-C.

set -uo pipefail
cd "$(dirname "$0")/.."

QUEUE="${1:-lab/active_queue.txt}"
POLL=60
SKIP_GPUS="${SKIP_GPUS:-2,4}"  # comma-separated list of dead/broken GPUs to skip

log() { echo "[$(date +%H:%M:%S)] $*"; }

get_busy_gpus() {
    nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader 2>/dev/null | \
        while IFS=, read -r pid bus; do
            gpu=$(nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader | grep "$bus" | cut -d, -f1 | tr -d ' ')
            echo "$gpu"
        done | sort -u
}

get_free_gpus() {
    local busy=$(get_busy_gpus | tr '\n' '|' | sed 's/|$//')
    local skip=$(echo "$SKIP_GPUS" | tr ',' '|')
    for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
        i=$(echo "$i" | tr -d ' ')
        # Skip broken GPUs
        if echo "$i" | grep -qE "^($skip)$"; then
            continue
        fi
        if [ -z "$busy" ] || ! echo "$i" | grep -qE "^($busy)$"; then
            echo "$i"
        fi
    done
}

next_job() {
    # Return first uncommented, non-DONE line from queue
    grep -v '^#' "$QUEUE" 2>/dev/null | grep -v 'DONE' | grep -v '^\s*$' | head -1
}

mark_done() {
    local name="$1"
    sed -i "s|^\(${name} .*\)|\1  # DONE|" "$QUEUE" 2>/dev/null
}

launch_job() {
    local gpu="$1" name="$2" steps="$3"
    shift 3
    local env_args="$*"

    log "Launching $name ($steps steps) on GPU $gpu"
    # Set env vars from queue line
    local env_cmd=""
    for arg in $env_args; do
        env_cmd="$env_cmd export $arg;"
    done

    bash -c "
        $env_cmd
        export CUDA_VISIBLE_DEVICES=$gpu
        lab/run_experiment.sh $name $steps > logs/${name}.txt 2>&1
    " &
    log "PID $! started for $name on GPU $gpu"
}

log "GPU monitor started. Queue: $QUEUE. Polling every ${POLL}s."

while true; do
    # Status report
    n_procs=$(ps aux | grep "python3 train_gpt" | grep -v grep | wc -l)
    free_gpus=$(get_free_gpus)
    n_free=$(echo "$free_gpus" | grep -c '[0-9]' || true)

    if [ "$n_free" -gt 0 ]; then
        log "$n_procs training procs, $n_free free GPUs: $free_gpus"

        # Try to fill free GPUs from queue
        for gpu in $free_gpus; do
            job_line=$(next_job)
            if [ -z "$job_line" ]; then
                log "Queue empty, no more jobs to launch."
                break
            fi
            name=$(echo "$job_line" | awk '{print $1}')
            steps=$(echo "$job_line" | awk '{print $2}')
            env_args=$(echo "$job_line" | awk '{for(i=3;i<=NF;i++) printf "%s ", $i}')

            # Check if already completed
            if [ -f "results/${name}/summary.json" ]; then
                mark_done "$name"
                continue
            fi

            launch_job "$gpu" "$name" "$steps" "$env_args"
            mark_done "$name"
            sleep 5  # let GPU claim memory before checking next
        done
    fi

    sleep "$POLL"
done
