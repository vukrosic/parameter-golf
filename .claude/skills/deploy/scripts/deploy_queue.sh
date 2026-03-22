#!/usr/bin/env bash
# Deploy a queue file to a remote GPU (no git required).
# Usage: bash deploy_queue.sh <gpu_name> <queue_file> [--push-code]
#
# --push-code: also sync train_gpt.py + infra/ before starting.
#              Use when local code has changed since last deploy.
#              Safe during active training (running processes already loaded the script).
#
# Situations:
#   Queue-only change:    deploy_queue.sh GPU queue.txt
#   Code + queue change:  deploy_queue.sh GPU queue.txt --push-code
#   Extend active run:    ssh into GPU and append to queues/active.txt manually

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
FLEET_SCRIPTS="$REPO_ROOT/.claude/skills/fleet/scripts"

GPU_NAME="${1:?Usage: deploy_queue.sh <gpu_name> <queue_file>}"
QUEUE_FILE="${2:?Usage: deploy_queue.sh <gpu_name> <queue_file>}"
PUSH_CODE=0
[ "${3:-}" = "--push-code" ] && PUSH_CODE=1

# Discover GPU credentials
GPU_INFO=$("$FLEET_SCRIPTS/discover_gpus.sh" | grep -i "^${GPU_NAME} " || true)
if [ -z "$GPU_INFO" ]; then
    echo "ERROR: GPU '$GPU_NAME' not found. Available GPUs:"
    "$FLEET_SCRIPTS/discover_gpus.sh" | awk '{print "  " $1}'
    exit 1
fi
PORT=$(echo "$GPU_INFO" | awk '{print $2}')
PASS=$(echo "$GPU_INFO" | awk '{print $3}')

source "$REPO_ROOT/infra/gpu_creds.sh"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o LogLevel=ERROR"
export SSHPASS="$PASS"

ssh_run() {
    sshpass -e ssh $SSH_OPTS -p "$PORT" root@"$HOST" "$@" 2>/dev/null
}

# Auto-detect remote repo path
REMOTE_REPO=$(ssh_run 'if [ -d /root/parameter-golf ]; then echo /root/parameter-golf; elif [ -d ~/parameter-golf ]; then echo ~/parameter-golf; else echo .; fi')
echo "Deploying $QUEUE_FILE → $GPU_NAME ($REMOTE_REPO)"

# Resolve queue file path
if [ -f "$QUEUE_FILE" ]; then
    QUEUE_PATH="$QUEUE_FILE"
elif [ -f "$REPO_ROOT/$QUEUE_FILE" ]; then
    QUEUE_PATH="$REPO_ROOT/$QUEUE_FILE"
elif [ -f "$REPO_ROOT/queues/$QUEUE_FILE" ]; then
    QUEUE_PATH="$REPO_ROOT/queues/$QUEUE_FILE"
else
    echo "ERROR: Queue file not found: $QUEUE_FILE"
    exit 1
fi
EXPERIMENTS=$(grep -v '^#' "$QUEUE_PATH" | grep -v '^$' | wc -l)
echo "  $EXPERIMENTS experiments in queue"

# Optionally push code
if [ $PUSH_CODE -eq 1 ]; then
    echo "  Pushing code..."
    bash "$REPO_ROOT/infra/push_code.sh" "$GPU_NAME"
fi

# Check if training already running
RUNNING=$(ssh_run 'pgrep -af train_gpt | grep -v grep | wc -l' || echo "0")
if [ "$RUNNING" -gt 0 ]; then
    echo "  WARNING: $RUNNING training process(es) already running on $GPU_NAME."
    echo "  Use /kill-exp to stop first, or append to the queue to extend the current run."
    exit 1
fi

# Write queue file via stdin (no scp/rsync needed)
echo "  Writing queue file..."
ssh_run "mkdir -p '$REMOTE_REPO/queues' '$REMOTE_REPO/logs' && cat > '$REMOTE_REPO/queues/active.txt'" \
    < "$QUEUE_PATH"

REMOTE_LINES=$(ssh_run "wc -l < '$REMOTE_REPO/queues/active.txt'" || echo "?")
echo "  Remote queue: $REMOTE_LINES lines"

# Start runner via bash -s (avoids multiline heredoc quoting issues)
echo "  Starting queue runner..."
sshpass -e ssh $SSH_OPTS -p "$PORT" root@"$HOST" "bash -s" 2>/dev/null <<SSHEOF
cd "$REMOTE_REPO"
nohup bash infra/run_queue.sh queues/active.txt >> logs/queue_runner_\$(date +%Y%m%d_%H%M).log 2>&1 &
echo "Runner PID: \$!"
SSHEOF

sleep 5
PROCS=$(ssh_run 'pgrep -af "run_queue\|train_gpt" | grep -v grep | wc -l' || echo "0")
if [ "$PROCS" -gt 0 ]; then
    echo "  Started — $PROCS process(es) active on $GPU_NAME"
    echo "  Monitor: /status $GPU_NAME"
else
    echo "  WARNING: runner may not have started — check logs on remote"
fi
