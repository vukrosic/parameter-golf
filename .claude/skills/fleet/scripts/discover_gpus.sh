#!/usr/bin/env bash
# Discover all GPUs from the merged fleet inventory.
# Output format: one line per GPU: <name> <port> <password>
# Primary source: /root/auto-research/auto_research.db when available.
# Fallback source: infra/gpu_creds.sh.
# Usage: source this or run it: bash .claude/skills/fleet/scripts/discover_gpus.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
python3 "$REPO_ROOT/infra/gpu_inventory.py" list
