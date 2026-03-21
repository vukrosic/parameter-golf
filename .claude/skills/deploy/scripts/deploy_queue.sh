#!/usr/bin/env bash
# Deploy a queue file to a remote GPU.
# Usage: bash deploy_queue.sh <gpu_name> <queue_file>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
FLEET_SCRIPTS="$REPO_ROOT/.claude/skills/fleet/scripts"

GPU_NAME="$1"
QUEUE_FILE="$2"

# Discover GPU credentials
GPU_INFO=$("$FLEET_SCRIPTS/discover_gpus.sh" | grep -i "^${GPU_NAME} " || true)
if [ -z "$GPU_INFO" ]; then
    echo "ERROR: GPU '$GPU_NAME' not found. Available GPUs:"
    "$FLEET_SCRIPTS/discover_gpus.sh" | awk '{print "  " $1}'
    exit 1
fi

PORT=$(echo "$GPU_INFO" | awk '{print $2}')
PASS=$(echo "$GPU_INFO" | awk '{print $3}')

echo "Deploying $QUEUE_FILE to $GPU_NAME (port $PORT)..."

# Ensure remote has latest code
echo "  Pulling latest code..."
bash "$FLEET_SCRIPTS/ssh_run.sh" "$PORT" "$PASS" "git pull origin lab 2>/dev/null" || true

# Check if something is already running
RUNNING=$(bash "$FLEET_SCRIPTS/ssh_run.sh" "$PORT" "$PASS" "ps aux | grep train_gpt.py | grep -v grep | head -1" || true)
if [ -n "$RUNNING" ]; then
    echo "  WARNING: An experiment is already running on $GPU_NAME:"
    echo "  $RUNNING"
    echo "  Use /kill-exp to stop it first."
    exit 1
fi

# Start the queue runner
echo "  Starting queue runner..."
bash "$FLEET_SCRIPTS/ssh_run.sh" "$PORT" "$PASS" \
    "nohup bash lab/run_queue.sh lab/$QUEUE_FILE > logs/queue_runner_$(date +%Y%m%d_%H%M).log 2>&1 &"

# Verify it started
sleep 2
CHECK=$(bash "$FLEET_SCRIPTS/ssh_run.sh" "$PORT" "$PASS" "ps aux | grep run_queue.sh | grep -v grep | head -1" || true)
if [ -n "$CHECK" ]; then
    echo "  Queue runner started successfully on $GPU_NAME."
else
    echo "  WARNING: Queue runner may not have started. Check logs on remote."
fi
