#!/usr/bin/env bash
# Kill running experiment on a specific GPU.
# Usage: bash kill_experiment.sh <gpu_name> [--force]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
FLEET_SCRIPTS="$REPO_ROOT/.claude/skills/fleet/scripts"

GPU_NAME="$1"
FORCE="${2:-}"

# Discover GPU credentials
GPU_INFO=$("$FLEET_SCRIPTS/discover_gpus.sh" | grep -i "^${GPU_NAME} " || true)
if [ -z "$GPU_INFO" ]; then
    echo "ERROR: GPU '$GPU_NAME' not found. Available GPUs:"
    "$FLEET_SCRIPTS/discover_gpus.sh" | awk '{print "  " $1}'
    exit 1
fi

PORT=$(echo "$GPU_INFO" | awk '{print $2}')
PASS=$(echo "$GPU_INFO" | awk '{print $3}')

# Check what's running
RUNNING=$(bash "$FLEET_SCRIPTS/ssh_run.sh" "$PORT" "$PASS" \
    'ps aux | grep train_gpt.py | grep -v grep | head -1' 2>/dev/null || true)

if [ -z "$RUNNING" ]; then
    echo "No experiment running on $GPU_NAME."
    exit 0
fi

echo "Running on $GPU_NAME:"
echo "  $RUNNING"

# Get current progress
PROGRESS=$(bash "$FLEET_SCRIPTS/ssh_run.sh" "$PORT" "$PASS" \
    'logf=$(ls -t logs/*.txt 2>/dev/null | head -1); [ -n "$logf" ] && tail -5 "$logf"' 2>/dev/null || echo "(no log)")
echo ""
echo "Recent log:"
echo "$PROGRESS"

if [ "$FORCE" != "--force" ]; then
    echo ""
    echo "Pass --force to kill, or use this skill interactively for confirmation."
    exit 0
fi

# Kill everything
echo ""
echo "Killing experiment..."
bash "$FLEET_SCRIPTS/ssh_run.sh" "$PORT" "$PASS" '
    pkill -f "train_gpt.py" 2>/dev/null
    pkill -f "run_queue.sh" 2>/dev/null
    pkill -f "run_experiment.sh" 2>/dev/null
    sleep 1
    remaining=$(ps aux | grep train_gpt.py | grep -v grep | wc -l)
    if [ "$remaining" -eq 0 ]; then
        echo "Experiment stopped. GPU is free."
    else
        echo "WARNING: Process still running. May need manual kill."
    fi
'
