#!/usr/bin/env bash
# Collect results from all remote GPUs immediately.
# Usage: bash collect_results.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
FLEET_SCRIPTS="$REPO_ROOT/.claude/skills/fleet/scripts"

cd "$REPO_ROOT"

echo "=== Collecting results from all GPUs @ $(date) ==="

# Count summary.json files before sync (flat + nested)
count_results() {
    { ls results/*/summary.json 2>/dev/null; ls results/*/*/summary.json 2>/dev/null; } | wc -l
}
BEFORE=$(count_results)

# Commit and push from each remote GPU
while IFS= read -r line; do
    name=$(echo "$line" | awk '{print $1}')
    port=$(echo "$line" | awk '{print $2}')
    pass=$(echo "$line" | awk '{print $3}')
    echo "--- $name ---"
    bash "$FLEET_SCRIPTS/ssh_run.sh" "$port" "$pass" '
        cd ~/parameter-golf 2>/dev/null || cd ~/param* 2>/dev/null || true
        # Stage all result summary/metadata files (flat and nested)
        git add results/*/summary.json results/*/metadata.json results/*/config.json 2>/dev/null || true
        git add results/*/*/summary.json results/*/*/metadata.json results/*/*/config.json 2>/dev/null || true
        if git diff --cached --quiet 2>/dev/null; then
            echo "  (nothing new)"
        else
            HOSTNAME_SHORT=$(hostname | cut -d. -f1)
            git commit -m "results($HOSTNAME_SHORT): auto-sync $(date +%Y-%m-%d_%H:%M)" 2>&1 | tail -1
            git push origin lab 2>&1 | tail -1 || echo "  PUSH FAILED"
        fi
    ' 2>/dev/null || echo "  OFFLINE — skipped"
done < <("$FLEET_SCRIPTS/discover_gpus.sh")

# Pull everything locally
echo "--- local pull ---"
git pull origin lab 2>&1 | tail -3 || echo "PULL FAILED"

# Commit and push any local results not yet committed
git add results/*/summary.json results/*/metadata.json results/*/config.json 2>/dev/null || true
git add results/*/*/summary.json results/*/*/metadata.json results/*/*/config.json 2>/dev/null || true
if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "results(local): auto-sync $(date +%Y-%m-%d_%H:%M)"
    git push origin lab 2>&1 | tail -1 || echo "LOCAL PUSH FAILED"
fi

# Pull on all remotes so they get the latest
while IFS= read -r line; do
    port=$(echo "$line" | awk '{print $2}')
    pass=$(echo "$line" | awk '{print $3}')
    bash "$FLEET_SCRIPTS/ssh_run.sh" "$port" "$pass" 'cd ~/parameter-golf 2>/dev/null || cd ~/param* 2>/dev/null; git pull origin lab 2>/dev/null' &
done < <("$FLEET_SCRIPTS/discover_gpus.sh")
wait

# Count after sync
AFTER=$(count_results)
NEW=$((AFTER - BEFORE))

echo ""
echo "=== Done: $NEW new experiment(s) synced ==="
if [ "$NEW" -gt 0 ]; then
    echo "New results (most recent first):"
    { ls -t results/*/summary.json 2>/dev/null; ls -t results/*/*/summary.json 2>/dev/null; } | head -"$NEW" | while read -r f; do
        expname=$(basename "$(dirname "$f")")
        bpb=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('last_eval',{}).get('val_bpb', d.get('val_bpb','?')))" 2>/dev/null || echo "?")
        echo "  $expname — val_bpb: $bpb"
    done
fi
