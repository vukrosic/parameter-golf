#!/usr/bin/env bash
# Deploy a single experiment to a remote GPU.
# Usage: bash deploy_single.sh <gpu_name> <exp_name> <steps> [ENV=val ...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
FLEET_SCRIPTS="$REPO_ROOT/.claude/skills/fleet/scripts"

GPU_NAME="$1"
EXP_NAME="$2"
STEPS="$3"
shift 3
ENV_VARS="$*"

# Discover GPU credentials
GPU_INFO=$("$FLEET_SCRIPTS/discover_gpus.sh" | grep -i "^${GPU_NAME} " || true)
if [ -z "$GPU_INFO" ]; then
    echo "ERROR: GPU '$GPU_NAME' not found. Available GPUs:"
    "$FLEET_SCRIPTS/discover_gpus.sh" | awk '{print "  " $1}'
    exit 1
fi

PORT=$(echo "$GPU_INFO" | awk '{print $2}')
PASS=$(echo "$GPU_INFO" | awk '{print $3}')

echo "Deploying $EXP_NAME ($STEPS steps) to $GPU_NAME..."

# Ensure remote has latest code
bash "$FLEET_SCRIPTS/ssh_run.sh" "$PORT" "$PASS" "git pull origin lab 2>/dev/null" || true

# Check if busy
RUNNING=$(bash "$FLEET_SCRIPTS/ssh_run.sh" "$PORT" "$PASS" "ps aux | grep train_gpt.py | grep -v grep | head -1" || true)
if [ -n "$RUNNING" ]; then
    echo "  WARNING: An experiment is already running on $GPU_NAME."
    echo "  $RUNNING"
    exit 1
fi

# Build env prefix
ENV_CMD=""
for ev in $ENV_VARS; do
    ENV_CMD="$ENV_CMD export $ev;"
done

# Start the experiment
echo "  Starting experiment..."
bash "$FLEET_SCRIPTS/ssh_run.sh" "$PORT" "$PASS" \
    "nohup bash -c '$ENV_CMD bash infra/run_experiment.sh $EXP_NAME $STEPS' > logs/${EXP_NAME}.txt 2>&1 &"

sleep 2
CHECK=$(bash "$FLEET_SCRIPTS/ssh_run.sh" "$PORT" "$PASS" "ps aux | grep train_gpt | grep -v grep | head -1" || true)
if [ -n "$CHECK" ]; then
    echo "  Experiment $EXP_NAME started on $GPU_NAME."
else
    echo "  WARNING: Experiment may not have started. Check logs."
fi
