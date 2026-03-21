#!/usr/bin/env bash
# Collect results from all remote GPUs immediately.
# Usage: bash collect_results.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
FLEET_SCRIPTS="$REPO_ROOT/.claude/skills/fleet/scripts"

cd "$REPO_ROOT"

echo "=== Collecting results from all GPUs @ $(date) ==="

# Count results before sync
BEFORE=$(ls -d results/*/submission.json 2>/dev/null | wc -l)

# Commit and push from each remote
"$FLEET_SCRIPTS/discover_gpus.sh" | while read -r name port pass; do
    echo "--- $name ---"
    bash "$FLEET_SCRIPTS/ssh_run.sh" "$port" "$pass" '
        git add results/*/submission.json results/*/config.json results/*/hparams.json results/*/README.md 2>/dev/null
        if git diff --cached --quiet 2>/dev/null; then
            echo "(nothing new)"
        else
            git commit -m "results('"$name"'): auto-sync $(date +%Y-%m-%d_%H:%M)"
            git push origin lab || echo "PUSH FAILED"
        fi
    ' 2>/dev/null || echo "  OFFLINE — skipped"
done

# Pull everything locally
echo "--- local pull ---"
git pull origin lab 2>/dev/null && echo "pulled OK" || echo "PULL FAILED"

# Push local results too
git add results/*/submission.json results/*/config.json results/*/hparams.json results/*/README.md 2>/dev/null
if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "results(local): auto-sync $(date +%Y-%m-%d_%H:%M)"
    git push origin lab || echo "LOCAL PUSH FAILED"
fi

# Pull on all remotes so they're in sync
"$FLEET_SCRIPTS/discover_gpus.sh" | while read -r name port pass; do
    bash "$FLEET_SCRIPTS/ssh_run.sh" "$port" "$pass" 'git pull origin lab 2>/dev/null' &
done
wait

# Count results after sync
AFTER=$(ls -d results/*/submission.json 2>/dev/null | wc -l)
NEW=$((AFTER - BEFORE))

echo ""
echo "=== Done: $NEW new experiment(s) synced ==="
if [ $NEW -gt 0 ]; then
    echo "New results:"
    # Show most recently modified submission.json files
    ls -t results/*/submission.json 2>/dev/null | head -"$NEW" | while read -r f; do
        name=$(echo "$f" | cut -d'/' -f2)
        bpb=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('val_bpb', '?'))" 2>/dev/null || echo "?")
        echo "  $name — val_bpb: $bpb"
    done
fi
