#!/usr/bin/env bash
# Collect results from all remote GPUs (no git required).
# Transport: tar | ssh pipe in both directions — zero dependencies on remote.
#
# Usage: bash collect_results.sh [GPU_NAME]
#   No arg: collect from all GPUs
#   GPU_NAME: collect from specific GPU only

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
FLEET_SCRIPTS="$REPO_ROOT/.claude/skills/fleet/scripts"

cd "$REPO_ROOT"

TARGET_GPU="${1:-}"

echo "=== Collecting results @ $(date) ==="

count_results() {
    { find results -maxdepth 2 -name 'summary.json' 2>/dev/null; \
      find results -maxdepth 3 -name 'summary.json' 2>/dev/null; } | sort -u | wc -l
}
BEFORE=$(count_results)

source "$REPO_ROOT/infra/gpu_creds.sh"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=12 -o LogLevel=ERROR"

collect_gpu() {
    local name="$1" port="$2" pass="$3"
    export SSHPASS="$pass"

    # Auto-detect remote repo path
    REMOTE_REPO=$(sshpass -e ssh $SSH_OPTS -p "$port" root@"$HOST" \
        'if [ -d /root/parameter-golf ]; then echo /root/parameter-golf; elif [ -d ~/parameter-golf ]; then echo ~/parameter-golf; else echo .; fi' 2>/dev/null) || { echo "  $name: OFFLINE"; return; }

    # Pull all result files via tar pipe (bash -s with cd ensures correct working dir)
    PULLED=$(sshpass -e ssh $SSH_OPTS -p "$port" root@"$HOST" "bash -s" 2>/dev/null <<SSHEOF | tar xzf - -C "$REPO_ROOT/" 2>/dev/null && echo ok || echo fail
cd "$REMOTE_REPO"
find results -name 'summary.json' -o -name 'metadata.json' -o -name 'config.json' | tar czf - -T - 2>/dev/null
SSHEOF
    )

    echo "  $name: collected ($PULLED)"
}

while IFS= read -r line; do
    name=$(echo "$line" | awk '{print $1}')
    port=$(echo "$line" | awk '{print $2}')
    pass=$(echo "$line" | awk '{print $3}')
    [ -n "$TARGET_GPU" ] && [ "$name" != "$TARGET_GPU" ] && continue
    echo "--- $name ---"
    collect_gpu "$name" "$port" "$pass"
done < <("$FLEET_SCRIPTS/discover_gpus.sh")

# Commit any new local results
git add results/ -A 2>/dev/null || true
if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "results: collect from fleet $(date +%Y-%m-%d_%H:%M)"
    echo "Committed new results locally."
fi

AFTER=$(count_results)
NEW=$((AFTER - BEFORE))
echo ""
echo "=== Done: $NEW new result(s) ==="
if [ "$NEW" -gt 0 ]; then
    { find results -maxdepth 2 -name 'summary.json'; \
      find results -maxdepth 3 -name 'summary.json'; } \
    | sort -u | xargs ls -t 2>/dev/null | head -"$NEW" | while read -r f; do
        exp=$(basename "$(dirname "$f")")
        bpb=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('last_eval',{}).get('val_bpb','?'))" 2>/dev/null || echo "?")
        echo "  $exp — $bpb bpb"
    done
fi
