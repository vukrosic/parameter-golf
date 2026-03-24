#!/usr/bin/env bash
# Research pipeline orchestrator
# Usage: bash research.sh [mode] [args...]
# Modes: decide, debate, queue, deepen, explore, status

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODE="${1:-interactive}"
shift 2>/dev/null || true

# ─── Helper: get next wave number ───────────────────────────────────────────
get_next_wave() {
    max=0
    for f in queues/wave_*_plan.md queues/wave_*.txt; do
        [ -f "$f" ] || continue
        num=$(echo "$f" | sed 's|queues/wave_||; s|_.*||')
        [ "${num:-0}" -gt "$max" ] 2>/dev/null && max=$num
    done
    echo $((max + 1))
}

# ─── Helper: latest result ───────────────────────────────────────────────────
latest_result() {
    ls -t results/*/summary.json results/*/summary.txt 2>/dev/null | head -1
}

# ─── Helper: current wave status ─────────────────────────────────────────────
wave_status() {
    if [ -f queues/active.txt ]; then
        echo "ACTIVE: $(cat queues/active.txt | grep -v '^#' | grep -v '^$' | head -3 | wc -l) jobs queued"
    else
        echo "NO active queue"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
# MODE: status — check current research state
# ═══════════════════════════════════════════════════════════════════════════
status_mode() {
    echo "═══ Research Pipeline Status ═══"
    echo ""
    wave_status
    echo ""

    # Latest results
    LATEST=$(latest_result)
    if [ -n "$LATEST" ]; then
        echo "Latest result: $(dirname "$LATEST")"
        head -5 "$LATEST"
    else
        echo "No results yet"
    fi
    echo ""

    # Wave history
    echo "Recent waves:"
    ls -t queues/wave_*_plan.md 2>/dev/null | head -5 | while read f; do
        status=$(grep "^Status:" "$f" | head -1 || echo "Status: unknown")
        echo "  $(basename "$f"): $status"
    done
    echo ""

    # Budget
    echo "Budget: (run /cost for details)"
}

# ═══════════════════════════════════════════════════════════════════════════
# MODE: decide — determine next action
# ═══════════════════════════════════════════════════════════════════════════
decide_mode() {
    echo "═══ Research Decision ═══"
    echo ""

    # Check for running experiments
    RUNNING=$(bash .claude/skills/fleet/scripts/fleet_status.sh 2>/dev/null | grep -c "training" || true)
    echo "Running experiments: $RUNNING"
    echo ""

    if [ "${RUNNING:-0}" -gt 0 ]; then
        echo "→ Experiments still running. Use /status to monitor."
        echo "→ Will prompt to decide after they complete."
        return
    fi

    # Check for completed wave with no action
    if [ -f queues/active.txt ]; then
        echo "Active queue found: queues/active.txt"
        echo "→ Deploy it with: /deploy queues/active.txt"
        return
    fi

    # No active work — time to decide
    echo "No active queue. Checking results for next decision..."

    # Get best recent result
    LATEST=$(latest_result)
    if [ -n "$LATEST" ]; then
        echo ""
        echo "Most recent: $(dirname "$LATEST")"
        python3 -c "
import json
d = json.load(open('$LATEST'))
bpb = d.get('val_bpb') or d.get('final_quant_eval', {}).get('val_bpb', '?')
print(f'  val_bpb: {bpb}')
" 2>/dev/null || true
    fi

    echo ""
    echo "═══ Decision Options ═══"
    echo ""
    echo "  /research mode=explore  → Start new idea category (Stage 1)"
    echo "  /research mode=deepen   → Follow up on winning result (Stage 2/3)"
    echo "  /research mode=debate    → Run debate first to decide direction"
    echo ""
    echo "Run /cost to check budget before deciding."
}

# ═══════════════════════════════════════════════════════════════════════════
# MODE: debate — run 3-agent debate, generate wave plan
# ═══════════════════════════════════════════════════════════════════════════
debate_mode() {
    TOPIC="${1:-Review recent results and decide next research direction}"
    WAVE_NUM=$(get_next_wave)

    echo "═══ Wave $WAVE_NUM Debate ═══"
    echo "Topic: $TOPIC"
    echo ""
    echo "Running full 3-round debate..."
    echo ""

    bash debate/debate.sh "$TOPIC" 3

    DEBATE_OUT=$(ls -td debate/rounds/*/ | head -1)
    echo ""
    echo "═══ Debate Complete ═══"
    echo "Output: $DEBATE_OUT"
    echo ""

    # Generate wave plan from synthesis
    echo "Generating wave_${WAVE_NUM}_plan.md..."
    python3 - <<'PYEOF'
import re, sys, os

repo = os.environ.get('REPO_ROOT', '/root/parameter-golf')
os.chdir(repo)

# Find latest debate round
import glob
debate_dirs = sorted(glob.glob('debate/rounds/*/'))
latest = debate_dirs[-1]
synth = os.path.join(latest, 'synthesis.md')

with open(synth) as f:
    content = f.read()

# Extract wave number from args or detect
wave_num = sys.argv[1] if len(sys.argv) > 1 else '1'

# Find queue entries (lines that look like run_experiment.sh commands)
queue_lines = []
for line in content.split('\n'):
    # Match infra/run_experiment.sh lines
    if 'run_experiment.sh' in line or 'infra/run_experiment.sh' in line:
        # Extract just the env vars and command
        clean = line.strip().lstrip('#').strip()
        queue_lines.append(clean)
    # Also capture bash code blocks
    elif line.startswith('```bash'):
        continue
    elif line.startswith('```'):
        break
    elif queue_lines and line.strip() and not line.startswith('#') and not line.startswith('##'):
        if '```' not in line:
            queue_lines.append(line.strip())

# Write queue file
queue_file = f'queues/wave_{wave_num}.txt'
with open(queue_file, 'w') as f:
    f.write(f"# Wave {wave_num}: auto-generated from debate synthesis\n")
    f.write(f"# Source: {latest}\n")
    f.write(f"# Edit this file before setting as active queue\n\n")
    for line in queue_lines:
        if line.strip() and not line.startswith('#'):
            f.write(line + '\n')

print(f"  ✓ Queue: {queue_file} ({len(queue_lines)} entries)")

# Write plan file
plan_file = f'queues/wave_{wave_num}_plan.md'
with open(plan_file, 'w') as f:
    f.write(f"""# Wave {wave_num} — Auto-generated from Debate

**Created:** {__import__('datetime').date.today().isoformat()}
**Source:** {latest}
**Status:** PENDING APPROVAL

---

## Consensus Summary
<!-- Edit this — auto-extracted from synthesis -->

## Key Disputes
<!-- Edit this — auto-extracted from synthesis -->

## Kill Criteria
- Improvement < 0.005 BPB at 500 steps → abort
- Size violation → redesign or abort
- Budget exceeded → stop

## Advancement Gate
Stage 2: 2000-4000 steps, 2 seeds, must beat current best
Stage 3: 13780 steps, must beat 1.2244 BPB

---

*Approved by:* _________  *Date:* _________
""")
print(f"  ✓ Plan: {plan_file}")
PYEOF
    $WAVE_NUM

    echo ""
    echo "Review and edit:"
    echo "  queues/wave_${WAVE_NUM}.txt       ← queue entries"
    echo "  queues/wave_${WAVE_NUM}_plan.md   ← wave plan"
    echo ""
    echo "To activate:"
    echo "  cp queues/wave_${WAVE_NUM}.txt queues/active.txt"
    echo "  /deploy queues/active.txt"
}

# ═══════════════════════════════════════════════════════════════════════════
# MODE: queue — generate queue file from wave plan
# ═══════════════════════════════════════════════════════════════════════════
queue_mode() {
    WAVE_NUM="${1:-$(get_next_wave)}"

    PLAN="queues/wave_${WAVE_NUM}_plan.md"
    QUEUE="queues/wave_${WAVE_NUM}.txt"

    if [ ! -f "$PLAN" ]; then
        echo "ERROR: $PLAN not found. Run /research mode=debate first."
        exit 1
    fi

    if [ ! -f "$QUEUE" ]; then
        echo "ERROR: $QUEUE not found. Edit the wave plan first."
        exit 1
    fi

    echo "=== Wave $WAVE_NUM Queue ==="
    echo ""
    cat "$QUEUE"
    echo ""
    echo "Plan: $PLAN"
    echo ""

    read -p "Set as active queue? (y/n): " confirm
    if [ "$confirm" = "y" ]; then
        cp "$QUEUE" queues/active.txt
        echo "✓ Set queues/active.txt"
        echo ""
        echo "Next: /deploy queues/active.txt to <GPU>"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
# MODE: deepen — run debate focused on deepening a winning direction
# ═══════════════════════════════════════════════════════════════════════════
deepen_mode() {
    echo "═══ Deepen Mode ═══"
    echo ""
    echo "Looking for Stage 1 winners to deepen..."

    # Find best recent validate/explore result
    BEST=$(ls -t results/validate/*/summary.json results/explore/*/summary.json 2>/dev/null | head -1)
    if [ -z "$BEST" ]; then
        echo "No validate/explore results found. Run Stage 1 first."
        echo "  /research mode=explore"
        return
    fi

    BEST_NAME=$(dirname "$BEST" | sed 's|results/||')
    echo "Best recent: $BEST_NAME"

    python3 -c "
import json
d = json.load(open('$BEST'))
bpb = d.get('val_bpb') or d.get('final_quant_eval', {}).get('val_bpb', '?')
print(f'  val_bpb: {bpb}')
" 2>/dev/null || true

    echo ""
    echo "Running Stage 2 debate focused on deepening $BEST_NAME..."
    debate_mode "Deepen $BEST_NAME: propose Stage 2 experiments (2000-4000 steps, 2 seeds, hyperparameter sweeps, stacking with known-good techniques)"
}

# ═══════════════════════════════════════════════════════════════════════════
# MODE: explore — run debate focused on new idea
# ═══════════════════════════════════════════════════════════════════════════
explore_mode() {
    echo "═══ Explore Mode ═══"
    echo ""

    # Check KNOWLEDGE.md for unexplored ideas
    echo "Checking KNOWLEDGE.md for unexplored Tier 1/2 ideas..."
    grep -A2 "Tier 1\|Tier 2" KNOWLEDGE.md 2>/dev/null | head -20 || true
    echo ""

    echo "Running Stage 1 explore debate..."
    debate_mode "Propose Stage 1 experiments (500-step quick screens, 4-8 ideas, new direction not yet explored)"
}

# ═══════════════════════════════════════════════════════════════════════════
# MODE: interactive — guide user through decision
# ═══════════════════════════════════════════════════════════════════════════
interactive_mode() {
    echo "═══ Research Pipeline ═══"
    echo ""
    echo "Current state:"
    status_mode | head -20
    echo ""
    echo "Available commands:"
    echo "  /research mode=decide    → Analyze results, decide next action"
    echo "  /research mode=status    → Show current wave and result status"
    echo "  /research mode=explore   → Start new idea (Stage 1)"
    echo "  /research mode=deepen    → Follow up winning result (Stage 2/3)"
    echo "  /research mode=debate     → Run 3-agent debate, generate wave plan"
    echo "  /research mode=queue     → Finalize and activate a wave queue"
    echo ""
    echo "Full pipeline: decide → debate → queue → deploy → collect → compare → post → repeat"
}

# ─── Dispatch ───────────────────────────────────────────────────────────────
case "$MODE" in
    status)     status_mode ;;
    decide)     decide_mode "$@" ;;
    debate)     debate_mode "$@" ;;
    queue)      queue_mode "$@" ;;
    deepen)     deepen_mode ;;
    explore)    explore_mode ;;
    interactive|"") interactive_mode ;;
    *)          echo "Unknown mode: $MODE" && exit 1 ;;
esac
