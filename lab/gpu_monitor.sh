#!/usr/bin/env bash
# GPU monitor: checks every 60s, launches queued jobs on free GPUs.
# Usage: lab/gpu_monitor.sh [queue_file]
# Runs forever. Kill with Ctrl-C.
#
# IMPORTANT: Uses CUDA device indices, NOT nvidia-smi indices.
# On this machine: CUDA 0=bus 0F (nv0), CUDA 1=bus 12 (nv1), CUDA 2=bus 8A (nv4).
# nvidia-smi GPUs 2 and 3 (bus 87, 89) are dead/invisible to CUDA.

set -uo pipefail
cd "$(dirname "$0")/.."

QUEUE="${1:-lab/active_queue.txt}"
POLL=60
LOCKDIR="/tmp/gpu_train_locks"
mkdir -p "$LOCKDIR"

# Working CUDA_VISIBLE_DEVICES values:
# CVD=0 → nv-smi 0 (bus 0F), CVD=1 → nv-smi 1 (bus 12),
# CVD=2 → nv-smi 3 (bus 89), CVD=3 → nv-smi 4 (bus 8A)
# nv-smi 2 (bus 87) is truly dead, not in CUDA enumeration.
CUDA_GPUS="0 1 2 3"

log() { echo "[$(date +%H:%M:%S)] $*"; }

gpu_has_training() {
    # Check if a GPU (CUDA index) has a running training process
    # Uses lock files: /tmp/gpu_train_locks/gpu_<N>.pid
    local gpu="$1"
    local lockfile="$LOCKDIR/gpu_${gpu}.pid"
    if [ -f "$lockfile" ]; then
        local pid=$(cat "$lockfile")
        if kill -0 "$pid" 2>/dev/null; then
            return 0  # process alive
        else
            # Stale lock — process died
            rm -f "$lockfile"
            return 1
        fi
    fi
    return 1  # no lock
}

get_free_gpus() {
    for gpu in $CUDA_GPUS; do
        if ! gpu_has_training "$gpu"; then
            echo "$gpu"
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
    local cuda_gpu="$1" name="$2" steps="$3"
    shift 3
    local env_args="$*"

    # SAFETY: double-check no process on this GPU
    if gpu_has_training "$cuda_gpu"; then
        log "ABORT: GPU $cuda_gpu already has a training process. Skipping $name."
        return 1
    fi

    log "Launching $name ($steps steps) on CUDA device $cuda_gpu"

    # Build env exports, filtering out comments
    local env_exports=""
    for arg in $env_args; do
        # Skip comments
        [[ "$arg" == "#"* ]] && break
        [[ "$arg" == *"="* ]] || continue
        env_exports="$env_exports $arg"
    done

    # Launch with proper env propagation
    (
        export CUDA_VISIBLE_DEVICES="$cuda_gpu"
        for kv in $env_exports; do
            export "$kv"
        done
        lab/run_experiment.sh "$name" "$steps" > "logs/${name}.txt" 2>&1
    ) &
    local pid=$!

    # Write lock file
    echo "$pid" > "$LOCKDIR/gpu_${cuda_gpu}.pid"
    log "PID $pid started for $name on CUDA device $cuda_gpu (lock: gpu_${cuda_gpu}.pid)"
}

# Register any already-running training processes
log "Scanning for existing training processes..."
for pid in $(pgrep -f "python3 train_gpt.py"); do
    cuda_dev=$(cat /proc/$pid/environ 2>/dev/null | tr '\0' '\n' | grep '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2)
    if [ -n "$cuda_dev" ]; then
        echo "$pid" > "$LOCKDIR/gpu_${cuda_dev}.pid"
        log "Found existing training PID $pid on CUDA device $cuda_dev"
    fi
done

log "GPU monitor started. Queue: $QUEUE. Polling every ${POLL}s. CUDA GPUs: $CUDA_GPUS"

while true; do
    # Clean stale locks
    for gpu in $CUDA_GPUS; do
        lockfile="$LOCKDIR/gpu_${gpu}.pid"
        if [ -f "$lockfile" ]; then
            pid=$(cat "$lockfile")
            if ! kill -0 "$pid" 2>/dev/null; then
                log "Cleaned stale lock for GPU $gpu (PID $pid dead)"
                rm -f "$lockfile"
            fi
        fi
    done

    free_gpus=$(get_free_gpus)
    n_free=$(echo "$free_gpus" | grep -c '[0-9]' || true)

    if [ "$n_free" -gt 0 ]; then
        n_procs=$(pgrep -fc "python3 train_gpt" || true)
        log "$n_procs training procs, $n_free free CUDA GPUs: $(echo $free_gpus | tr '\n' ' ')"

        for gpu in $free_gpus; do
            job_line=$(next_job)
            if [ -z "$job_line" ]; then
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
            sleep 10  # let GPU claim memory before checking next
        done
    fi

    sleep "$POLL"
done
