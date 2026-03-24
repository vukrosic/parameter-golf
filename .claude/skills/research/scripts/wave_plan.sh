#!/usr/bin/env bash
# Generate wave plan and queue files from debate synthesis output.
# Usage: bash wave_plan.sh <debate_timestamp_dir> <wave_number>
#
# Example:
#   bash wave_plan.sh debate/rounds/20260322_143000/ 30

set -euo pipefail

DEBATE_DIR="${1:?Usage: bash wave_plan.sh <debate_dir> <wave_number>}"
WAVE_NUM="${2:?Usage: bash wave_plan.sh <debate_dir> <wave_number>}"
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

SYNTHESIS="$DEBATE_DIR/synthesis.md"
if [ ! -f "$SYNTHESIS" ]; then
    echo "ERROR: synthesis.md not found at $SYNTHESIS"
    exit 1
fi

PLAN_FILE="queues/wave_${WAVE_NUM}_plan.md"
QUEUE_FILE="queues/wave_${WAVE_NUM}.txt"

echo "Generating wave_${WAVE_NUM} plan and queue from debate output..."
echo "  Debate: $DEBATE_DIR"
echo "  Plan:   $PLAN_FILE"
echo "  Queue:  $QUEUE_FILE"
echo ""

# Extract queue entries from synthesis markdown
python3 - <<PYEOF
import sys, re

synth_path = sys.argv[1]
wave_num = sys.argv[2]
with open(synth_path) as f:
    content = f.read()

queue_entries = []
in_queue = False
for line in content.split('\n'):
    # Detect queue section
    if re.search(r'[Ee]xperiment\s+[Qq]ueue|[Qq]ueue:\s*$', line):
        in_queue = True
        continue
    if in_queue and line.startswith('## '):
        break
    # Skip code fences
    if '```' in line:
        in_queue = not in_queue and in_queue  # toggle
        continue
    if in_queue and line.strip():
        # Clean the line (remove leading # or bash markers)
        clean = line.strip().lstrip('#').strip()
        if clean and not clean.startswith('##'):
            queue_entries.append(clean)

# Write queue file
queue_file = f'queues/wave_{wave_num}.txt'
with open(queue_file, 'w') as f:
    f.write(f"# Wave {wave_num}: auto-generated from debate synthesis\n")
    f.write(f"# Source: {synth_path}\n")
    f.write(f"# Edit before setting as active queue\n\n")
    for entry in queue_entries:
        f.write(entry + '\n')

print(f"  Queue: {queue_file} ({len(queue_entries)} entries)")
if not queue_entries:
    print("  WARNING: No queue entries extracted — check synthesis format")
PYEOF
"$SYNTHESIS" "$WAVE_NUM"

# Write plan file
cat > "$PLAN_FILE" <<PLANEOF
# Wave $WAVE_NUM — $(date +%Y-%m-%d)

**Source:** $DEBATE_DIR
**Status:** PENDING APPROVAL

---

## Context Summary
<!-- Auto-generated from debate synthesis. Edit before approving. -->

## Decision
$(grep -A 5 "^## Consensus" "$SYNTHESIS" 2>/dev/null | head -10 || echo "See synthesis.md")

## Key Disputes
$(grep -A 10 "^## Key Disputes" "$SYNTHESIS" 2>/dev/null | head -15 || echo "See synthesis.md")

## Kill Criteria
- Improvement < 0.005 BPB at 500 steps → abort remaining
- Size violation → redesign or abort
- Budget exceeded → stop adding experiments

## Advancement Gate
Stage 2: 2000-4000 steps, 2 seeds, must beat current best at matching steps
Stage 3: 13780 steps, 2 seeds, must beat 1.2244 BPB

## Budget Estimate
<!-- Fill in before approving -->
| Experiment | Steps | Est. Cost |
|---|---|---|
| | | |
| **Total** | | ~$ |

---

*Approved by:* _________  *Date:* _________
PLANEOF

echo "  ✓ Plan:   $PLAN_FILE"
echo ""
echo "Next steps:"
echo "  1. Review and edit: queues/wave_${WAVE_NUM}.txt"
echo "  2. Edit kill criteria and budget in: queues/wave_${WAVE_NUM}_plan.md"
echo "  3. Remove PENDING APPROVAL when ready"
echo "  4. Activate: cp queues/wave_${WAVE_NUM}.txt queues/active.txt"
echo "  5. Deploy: /deploy queues/active.txt to <GPU>"
