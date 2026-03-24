#!/usr/bin/env bash
# Draft X/Twitter posts from experiment results or debate outputs.
# Usage: bash post.sh [mode] [experiment_name]
#
# Modes:
#   stage1      — Post Stage 1 (Explore) results
#   stage2      — Post Stage 2 (Validate) results
#   stage3      — Post Stage 3 (Full run) results
#   latest      — Auto-detect latest result and draft appropriate post
#   from=<name> — Draft from specific experiment

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-latest}"
shift 2>/dev/null || true

# ─── Helper: extract val_bpb from summary ──────────────────────────────────
extract_bpb() {
    python3 -c "
import json, sys
f = sys.argv[1]
d = json.load(open(f))
bpb = d.get('val_bpb') or d.get('final_quant_eval', {}).get('val_bpb', None)
if bpb:
    print(f'{bpb:.4f}')
else:
    print('?.????')
" "$1" 2>/dev/null || echo "?.????"
}

# ─── Helper: extract postable info from result ───────────────────────────────
get_result_info() {
    python3 - <<'PYEOF'
import json, sys, os, glob

name = sys.argv[1] if len(sys.argv) > 1 else ''
base = f'results/{name}'

# Find summary file
for f in [f'{base}/summary.json', f'{base}/submission.json']:
    if os.path.exists(f):
        d = json.load(open(f))
        break
else:
    print('???')
    sys.exit(1)

val_bpb = d.get('val_bpb') or d.get('final_quant_eval', {}).get('val_bpb', '?')
quant_bpb = d.get('final_quant_eval', {}).get('int8_zlib_bpb', None) or d.get('int8_zlib_bpb', None)
steps = d.get('steps', d.get('total_steps', '?'))
params = d.get('params', '?')
size_mb = d.get('size_mb', '?')

print(f"NAME={name}")
print(f"VAL_BPB={val_bpb}")
print(f"QUANT_BPB={quant_bpb}")
print(f"STEPS={steps}")
print(f"PARAMS={params}")
print(f"SIZE_MB={size_mb}")
PYEOF
}

# ─── Mode: latest ───────────────────────────────────────────────────────────
draft_latest() {
    # Find latest completed result
    RESULT_DIR=$(ls -t results/*/summary.json results/*/submission.json 2>/dev/null | grep -v "full/" | head -1 | xargs dirname 2>/dev/null || true)
    if [ -z "$RESULT_DIR" ]; then
        echo "No results found."
        return
    fi
    RESULT_NAME=$(basename "$RESULT_DIR")
    echo "Latest result: $RESULT_NAME"
    echo ""

    # Determine stage from path
    case "$RESULT_DIR" in
        */explore/*) draft_stage1 "$RESULT_NAME" ;;
        */validate/*) draft_stage2 "$RESULT_NAME" ;;
        */full/*) draft_stage3 "$RESULT_NAME" ;;
        *) draft_stage2 "$RESULT_NAME" ;;
    esac
}

# ─── Mode: stage1 ──────────────────────────────────────────────────────────
draft_stage1() {
    NAME="${1:-$(ls -t results/explore/*/summary.json 2>/dev/null | head -1 | xargs dirname | xargs basename)}"
    [ -z "$NAME" ] && echo "No explore results found." && return

    BPB=$(extract_bpb "results/$NAME/summary.json")
    DELTA=$(python3 -c "print(f'{float(1.4793) - float(\"$BPB\"):.4f}')" 2>/dev/null || echo "?.???")
    STEPS=$(python3 -c "import json; print(json.load(open('results/$NAME/summary.json')).get('steps',500))" 2>/dev/null || echo "500")

    # Count how many were in this explore wave
    COUNT=$(ls -d results/explore/${NAME%_*}_* 2>/dev/null | wc -l || echo "?")

    python3 -c "
import json
d = json.load(open('results/$NAME/summary.json'))
params = d.get('params', '?')
size = d.get('size_mb', '?')
" 2>/dev/null || true

    echo "🚨 **Parameter Golf Experiment Results**

Quick screen (${STEPS} steps)

🎯 Best: **$NAME** — $BPB BPB (Δ${DELTA} vs baseline)

Baseline: 1.4793 BPB

📊 Next: Moving to Stage 2 validation

#ParameterGolf #ML"
}

# ─── Mode: stage2 ──────────────────────────────────────────────────────────
draft_stage2() {
    # Find best validate result
    NAME="${1:-$(ls -t results/validate/*/summary.json 2>/dev/null | head -1 | xargs dirname | xargs basename)}"
    [ -z "$NAME" ] && echo "No validate results found." && return

    BPB=$(extract_bpb "results/$NAME/summary.json")
    STEPS=$(python3 -c "import json; print(json.load(open('results/$NAME/summary.json')).get('steps',4000))" 2>/dev/null || echo "4000")
    KNOWN_BEST=1.3637  # MoE4e + bn128_untied at 4000 steps

    DELTA_KNOWN=$(python3 -c "print(f'{float(\"$KNOWN_BEST\") - float(\"$BPB\"):.4f}')" 2>/dev/null || echo "?.???")
    DELTA_BASELINE=$(python3 -c "print(f'{float(1.4793) - float(\"$BPB\"):.4f}')" 2>/dev/null || echo "?.???")

    # Try to find seed variant
    SEED=$(echo "$NAME" | grep -o 's[0-9]*' | head -1 || echo "")
    OTHER_SEED=""
    if [ -n "$SEED" ]; then
        OTHER=$(echo "$NAME" | sed "s/$SEED//")
        for s in s42 s1337; do
            [ "$s" = "$SEED" ] && continue
            [ -f "results/validate/${OTHER}${s}/summary.json" ] && OTHER_SEED=$s && break
        done
    fi

    echo "📈 **Parameter Golf — Validation Results**

**$NAME** at ${STEPS} steps

Pre-quant BPB: $BPB
vs baseline (1.4793): Δ${DELTA_BASELINE}
vs best known ($KNOWN_BEST): Δ${DELTA_KNOWN}

$(if [ -n "$OTHER_SEED" ]; then
BPB2=$(extract_bpb "results/validate/${OTHER}${OTHER_SEED}/summary.json")
echo "Seed comparison: ${SEED#$SEED}=${BPB}, ${OTHER_SEED}=${BPB2}"
fi)

$(if python3 -c "exit(0 if float('$BPB') < float('$KNOWN_BEST') else 1)" 2>/dev/null; then
echo "✅ Promising — scaling to full run"
else
echo "⚠️ Did not beat best known — more validation needed"
fi)

#ParameterGolf #ML"
}

# ─── Mode: stage3 ──────────────────────────────────────────────────────────
draft_stage3() {
    NAME="${1:-$(ls -t results/full/*/summary.json results/full/*/submission.json 2>/dev/null | head -1 | xargs dirname | xargs basename)}"
    [ -z "$NAME" ] && echo "No full results found." && return

    python3 - <<PYEOF
import json
d = json.load(open('results/$NAME/summary.json'))
val_bpb = d.get('val_bpb') or d.get('final_quant_eval', {}).get('val_bpb', '?')
quant_bpb = d.get('final_quant_eval', {}).get('int8_zlib_bpb', None) or d.get('int8_zlib_bpb', '?')
params = d.get('params', '?')
size_mb = d.get('size_mb', '?')
steps = d.get('steps', d.get('total_steps', '?'))

leaderboard = 1.2244
try:
    delta = float(quant_bpb) - leaderboard
    delta_str = f'{delta:+.4f}' if delta != '?' else '?.????'
except:
    delta_str = '?.????'

print(f"""🏁 **Parameter Golf — Full Run Complete**

Config: $NAME
Steps: {steps}
Params: {params}
Size: {size_mb} MB

Pre-quant:  {val_bpb}
Post-quant: {quant_bpb}

Leaderboard baseline: {leaderboard}
Our result:           {quant_bpb} ({delta_str})

#ParameterGolf #ML""")
PYEOF
}

# ─── Dispatch ───────────────────────────────────────────────────────────────
case "$MODE" in
    stage1)  draft_stage1 "$@" ;;
    stage2)  draft_stage2 "$@" ;;
    stage3)  draft_stage3 "$@" ;;
    latest)  draft_latest ;;
    from=*)  NAME="${MODE#from=}"; draft_stage2 "$NAME" ;;
    *)       echo "Usage: post.sh [stage1|stage2|stage3|latest|from=<name>]" ;;
esac
