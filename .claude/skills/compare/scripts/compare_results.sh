#!/usr/bin/env bash
# Compare experiment results. Show ranked table or side-by-side diff.
# Usage: bash compare_results.sh [exp1 exp2] [--pattern glob]
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

if [ $# -eq 0 ]; then
    # Default: ranked leaderboard of all results
    echo "=== Experiment Leaderboard (ranked by val_bpb) ==="
    echo ""
    printf "%-4s %-40s %-10s %-10s %-8s\n" "Rank" "Experiment" "val_bpb" "val_loss" "Steps"
    printf "%-4s %-40s %-10s %-10s %-8s\n" "----" "----------" "-------" "--------" "-----"

    # Collect all results with val_bpb
    python3 -c "
import json, os, glob

results = []
for f in glob.glob('results/*/submission.json'):
    try:
        d = json.load(open(f))
        name = os.path.basename(os.path.dirname(f))
        bpb = d.get('val_bpb', d.get('final_val_bpb', None))
        loss = d.get('val_loss', d.get('final_val_loss', '?'))
        steps = d.get('steps', d.get('iterations', '?'))
        if bpb is not None:
            results.append((float(bpb), name, loss, steps))
    except:
        pass

results.sort()
for i, (bpb, name, loss, steps) in enumerate(results, 1):
    print(f'{i:<4} {name:<40} {bpb:<10.4f} {str(loss):<10} {str(steps):<8}')

if not results:
    print('No results found. Run experiments first.')
" 2>/dev/null

elif [ $# -eq 2 ] && [[ "$1" != --* ]]; then
    # Side-by-side comparison of two experiments
    EXP1="$1"
    EXP2="$2"
    echo "=== Comparing: $EXP1 vs $EXP2 ==="
    echo ""

    python3 -c "
import json, os

def load(name):
    d = {}
    for f in ['submission.json', 'config.json', 'hparams.json']:
        p = f'results/{name}/{f}'
        if os.path.exists(p):
            d.update(json.load(open(p)))
    return d

a = load('$EXP1')
b = load('$EXP2')
all_keys = sorted(set(list(a.keys()) + list(b.keys())))

print(f'{\"Config\":<30} {\"$EXP1\":<25} {\"$EXP2\":<25} {\"Delta\":<15}')
print(f'{\"---\":<30} {\"---\":<25} {\"---\":<25} {\"---\":<15}')

for k in all_keys:
    va = a.get(k, '-')
    vb = b.get(k, '-')
    delta = ''
    if va != vb:
        try:
            d = float(vb) - float(va)
            delta = f'{d:+.4f}' if abs(d) < 100 else f'{d:+.1f}'
        except:
            delta = 'changed'
    else:
        delta = 'same'
    print(f'{k:<30} {str(va):<25} {str(vb):<25} {delta:<15}')
" 2>/dev/null

else
    echo "Usage: compare_results.sh [exp1 exp2]"
    echo "  No args: show ranked leaderboard"
    echo "  Two args: side-by-side comparison"
fi
