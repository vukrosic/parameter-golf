#!/usr/bin/env bash
# Quick fleet status: check each GPU for nvidia-smi + running experiment.
# Usage: bash fleet_status.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source "$REPO_ROOT/lab/gpu_creds.sh"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5 -o LogLevel=ERROR"

ssh_run() {
    local port="$1" pass="$2" cmd="$3"
    sshpass -p "$pass" ssh $SSH_OPTS -p "$port" root@"$HOST" "cd /root/parameter-golf && $cmd" 2>/dev/null
}

echo "=== GPU Fleet Status @ $(date '+%Y-%m-%d %H:%M:%S') ==="
echo ""
printf "%-15s %-8s %-6s %-6s %-30s %-10s\n" "GPU" "STATUS" "UTIL%" "TEMP" "EXPERIMENT" "STEP"
printf "%-15s %-8s %-6s %-6s %-30s %-10s\n" "---" "------" "-----" "----" "----------" "----"

# Discover GPUs dynamically
"$SCRIPT_DIR/discover_gpus.sh" | while read -r name port pass; do
    # Fetch stats in one SSH call
    info=$(ssh_run "$port" "$pass" '
        gpu=$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "?,?")
        exp=$(ps aux | grep train_gpt.py | grep -v grep | head -1 | grep -oP "RUN_ID=\K\S+" || echo "")
        if [ -z "$exp" ]; then
            exp=$(ps aux | grep run_experiment.sh | grep -v grep | head -1 | awk "{print \$NF}" | head -c 30 || echo "")
        fi
        step=""
        if [ -n "$exp" ]; then
            logf="logs/${exp}.txt"
            if [ -f "$logf" ]; then
                step=$(grep -oP "step \K[0-9]+(/[0-9]+)?" "$logf" 2>/dev/null | tail -1)
            fi
        fi
        echo "$gpu|$exp|$step"
    ' 2>/dev/null) || info="OFFLINE"

    if [ "$info" = "OFFLINE" ]; then
        printf "%-15s %-8s %-6s %-6s %-30s %-10s\n" "$name" "OFFLINE" "-" "-" "-" "-"
    else
        util=$(echo "$info" | cut -d'|' -f1 | cut -d',' -f1 | tr -d ' ')
        temp=$(echo "$info" | cut -d'|' -f1 | cut -d',' -f2 | tr -d ' ')
        exp=$(echo "$info" | cut -d'|' -f2)
        step=$(echo "$info" | cut -d'|' -f3)
        status="IDLE"
        [ -n "$exp" ] && status="RUNNING"
        [ -z "$exp" ] && exp="-"
        [ -z "$step" ] && step="-"
        printf "%-15s %-8s %-6s %-6s %-30s %-10s\n" "$name" "$status" "${util}%" "${temp}C" "$exp" "$step"
    fi
done

echo ""
echo "=== End ==="
