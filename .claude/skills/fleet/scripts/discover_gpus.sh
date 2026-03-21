#!/usr/bin/env bash
# Discover all GPUs from gpu_creds.sh dynamically.
# Output format: one line per GPU: <name> <port> <password>
# Usage: source this or run it: bash .claude/skills/fleet/scripts/discover_gpus.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
CREDS_FILE="$REPO_ROOT/infra/gpu_creds.sh"

if [ ! -f "$CREDS_FILE" ]; then
    echo "ERROR: $CREDS_FILE not found. Run /add-gpu to set up credentials." >&2
    exit 1
fi

source "$CREDS_FILE"

# Extract GPU names by finding all GPU_*_PORT variables
grep -oP 'GPU_\K[A-Z0-9_]+(?=_PORT=)' "$CREDS_FILE" | while read -r name; do
    port_var="GPU_${name}_PORT"
    pass_var="GPU_${name}_PASS"
    port="${!port_var}"
    pass="${!pass_var}"
    if [ -n "$port" ] && [ -n "$pass" ]; then
        echo "$name $port $pass"
    fi
done
