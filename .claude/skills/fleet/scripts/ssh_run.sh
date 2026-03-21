#!/usr/bin/env bash
# SSH into a GPU and run a command. Shared helper for all skills.
# Usage: bash ssh_run.sh <port> <password> <command>
# Reads HOST from gpu_creds.sh automatically.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
source "$REPO_ROOT/lab/gpu_creds.sh"

PORT="$1"
PASS="$2"
shift 2
CMD="$*"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o LogLevel=ERROR"
sshpass -p "$PASS" ssh $SSH_OPTS -p "$PORT" root@"$HOST" "cd /root/parameter-golf && $CMD" 2>/dev/null
