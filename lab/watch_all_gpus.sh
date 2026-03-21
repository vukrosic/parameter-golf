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
    local out
    out=$(sshpass -p "$pass" ssh $SSH_OPTS -p "$port" root@$HOST \
        "tail -1 $REPO/logs/*.txt 2>/dev/null | grep -oP 'step:\d+/\d+.*step_avg:\S+' | tail -1; echo '|||'; nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null" 2>/dev/null)
    local step_line=$(echo "$out" | head -1)
    local gpu_info=$(echo "$out" | tail -1)
    [ "$step_line" = "|||" ] || [ -z "$step_line" ] && step_line="(idle)"
    printf "%-18s | %-55s | %s\n" "$label" "$step_line" "${gpu_info:-offline}"
}

local_status() {
    local step_line gpu_info
    step_line=$(tail -1 $REPO/logs/*.txt 2>/dev/null | grep -oP 'step:\d+/\d+.*step_avg:\S+' | tail -1)
    gpu_info=$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null)
    [ -z "$step_line" ] && step_line="(idle)"
    printf "%-18s | %-55s | %s\n" "LOCAL 3090" "$step_line" "${gpu_info:-N/A}"
}

echo "=== GPU Fleet @ $(date '+%H:%M:%S') ==="
echo "------------------+---------------------------------------------------------+--------------------"
printf "%-18s | %-55s | %s\n" "GPU" "Progress" "Util%, Temp°C, Watts"
echo "------------------+---------------------------------------------------------+--------------------"
local_status &
remote_status "REMOTE 3090" $GPU_REMOTE3090_PORT "$GPU_REMOTE3090_PASS" &
remote_status "L40S"        $GPU_L40S_PORT       "$GPU_L40S_PASS" &
remote_status "RTX 5090"    $GPU_5090_PORT       "$GPU_5090_PASS" &
wait
echo "------------------+---------------------------------------------------------+--------------------"
