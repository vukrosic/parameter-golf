#!/usr/bin/env bash
# Test connectivity to a new GPU and report its specs.
# Usage: bash add_gpu.sh <port> <password>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source "$REPO_ROOT/lab/gpu_creds.sh"

PORT="$1"
PASS="$2"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o LogLevel=ERROR"

echo "Testing connectivity to port $PORT..."
GPU_INFO=$(sshpass -p "$PASS" ssh $SSH_OPTS -p "$PORT" root@"$HOST" \
    "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader" 2>/dev/null) || {
    echo "ERROR: Cannot connect to port $PORT. Check host/port/password."
    exit 1
}

echo "Connected! GPU specs:"
echo "  $GPU_INFO"

# Quick disk check
DISK=$(sshpass -p "$PASS" ssh $SSH_OPTS -p "$PORT" root@"$HOST" "df -h /root | tail -1" 2>/dev/null || echo "unknown")
echo "  Disk: $DISK"

echo ""
echo "Connection verified. Ready to add to fleet."
