#!/bin/bash
# Watch training progress across all 4 GPUs
# Usage: watch -n 30 bash lab/watch_all_gpus.sh
# Credentials loaded from lab/gpu_creds.sh (gitignored)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/gpu_creds.sh"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5 -o LogLevel=ERROR"
REPO="/root/parameter-golf"

remote_status() {
    local label="$1" port="$2" pass="$3"
    local env_info step_info gpu_info
    env_info=$(sshpass -p "$pass" ssh $SSH_OPTS -p "$port" root@$HOST \
        'xargs -0 -L1 < /proc/$(pgrep -f train_gpt | head -1)/environ 2>/dev/null' 2>/dev/null | grep -E 'RUN_ID|ITERATIONS' | sort)
    local run_id=$(echo "$env_info" | grep RUN_ID | cut -d= -f2)
    local iters=$(echo "$env_info" | grep ITERATIONS | cut -d= -f2)
    gpu_info=$(sshpass -p "$pass" ssh $SSH_OPTS -p "$port" root@$HOST \
        "nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null" 2>/dev/null)
    if [ -z "$run_id" ]; then
        run_id="(idle)"
    else
        run_id="$run_id ($iters steps)"
    fi
    printf "%-22s | %-45s | %s\n" "$label" "$run_id" "${gpu_info:-offline}"
}

local_status() {
    local env_info run_id iters gpu_info
    env_info=$(xargs -0 -L1 < /proc/$(pgrep -f train_gpt | head -1)/environ 2>/dev/null | grep -E 'RUN_ID|ITERATIONS' | sort)
    run_id=$(echo "$env_info" | grep RUN_ID | cut -d= -f2)
    iters=$(echo "$env_info" | grep ITERATIONS | cut -d= -f2)
    gpu_info=$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null)
    if [ -z "$run_id" ]; then
        run_id="(idle)"
    else
        run_id="$run_id ($iters steps)"
    fi
    printf "%-22s | %-45s | %s\n" "LOCAL 3090" "$run_id" "${gpu_info:-N/A}"
}

echo "=== GPU Fleet Status @ $(date '+%H:%M:%S') ==="
echo "----------------------+-----------------------------------------------+------------------"
printf "%-22s | %-45s | %s\n" "GPU" "Experiment" "Util%, Temp°C, Watts"
echo "----------------------+-----------------------------------------------+------------------"
local_status &
remote_status "REMOTE 3090 (:$GPU_REMOTE3090_PORT)" $GPU_REMOTE3090_PORT "$GPU_REMOTE3090_PASS" &
remote_status "L40S (:$GPU_L40S_PORT)"               $GPU_L40S_PORT       "$GPU_L40S_PASS" &
remote_status "RTX 5090 (:$GPU_5090_PORT)"            $GPU_5090_PORT       "$GPU_5090_PASS" &
wait
echo "----------------------+-----------------------------------------------+------------------"
