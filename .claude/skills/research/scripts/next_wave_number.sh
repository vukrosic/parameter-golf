#!/usr/bin/env bash
# Find the next wave number by scanning queues/ for existing wave_NN_plan.md files
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

max=0
for f in queues/wave_*_plan.md queues/wave_*.txt; do
    if [ -f "$f" ]; then
        num=$(echo "$f" | sed 's|queues/wave_||; s|_.*||')
        if [ "$num" -gt "$max" ] 2>/dev/null; then
            max=$num
        fi
    fi
done
echo $((max + 1))
