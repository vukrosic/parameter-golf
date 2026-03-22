#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# /wave — Unified research loop orchestrator
# Usage: bash wave.sh [status|plan|approve|results|pivot] [options...]
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-status}"
shift 2>/dev/null || true

# ─── Helpers ──────────────────────────────────────────────────────────

get_next_wave() {
    local max=0
    for f in queues/wave_*_plan.md queues/wave_*.txt; do
        [ -f "$f" ] || continue
        local num
        num=$(echo "$f" | sed 's|queues/wave_||; s|_.*||; s|\..*||')
        [ "${num:-0}" -gt "$max" ] 2>/dev/null && max=$num
    done
    echo $((max + 1))
}

get_current_wave() {
    local max=0
    for f in queues/wave_*_plan.md queues/wave_*.txt; do
        [ -f "$f" ] || continue
        local num
        num=$(echo "$f" | sed 's|queues/wave_||; s|_.*||; s|\..*||')
        [ "${num:-0}" -gt "$max" ] 2>/dev/null && max=$num
    done
    echo "$max"
}

latest_result_dir() {
    ls -td results/*/  2>/dev/null | head -1
}

detect_phase() {
    # What phase are the current/latest experiments in?
    if [ -f queues/active.txt ]; then
        local first_name
        first_name=$(grep -v '^#' queues/active.txt | grep -v '^$' | head -1 | awk '{print $1}')
        case "$first_name" in
            explore*) echo "explore" ;;
            validate*) echo "validate" ;;
            full*) echo "full" ;;
            *) echo "unknown" ;;
        esac
    else
        # Check latest results
        local latest
        latest=$(ls -t results/*/summary.json results/*/summary.txt 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            local dir
            dir=$(dirname "$latest" | sed 's|results/||')
            case "$dir" in
                explore/*) echo "explore" ;;
                validate/*) echo "validate" ;;
                full/*) echo "full" ;;
                *) echo "unknown" ;;
            esac
        else
            echo "none"
        fi
    fi
}

detect_debate_type() {
    # Auto-detect what kind of debate to run
    local phase
    phase=$(detect_phase)

    # Check budget
    local budget_low=false
    if command -v python3 &>/dev/null; then
        # Simple heuristic: check if /cost skill exists and budget is low
        # For now, default to direction unless we have winners
        :
    fi

    # Check for consecutive stale waves
    local stale_count=0
    for plan in $(ls -t queues/wave_*_plan.md 2>/dev/null | head -5); do
        if grep -q "no improvement\|no winner\|pivot\|failed" "$plan" 2>/dev/null; then
            stale_count=$((stale_count + 1))
        else
            break
        fi
    done

    if [ "$stale_count" -ge 3 ]; then
        echo "pivot"
        return
    fi

    case "$phase" in
        explore)
            # If we just ran explore, we need a scale debate to decide validation
            echo "scale"
            ;;
        validate)
            # If we just validated, scale debate for full run decision
            echo "scale"
            ;;
        full)
            # After a full run, direction debate for next research
            echo "direction"
            ;;
        *)
            # Default: direction (new exploration)
            echo "direction"
            ;;
    esac
}

agents_for_type() {
    local debate_type="$1"
    case "$debate_type" in
        direction) echo "architect explorer skeptic" ;;
        scale)     echo "architect challenger optimizer" ;;
        pivot)     echo "architect skeptic explorer challenger optimizer" ;;
        *)         echo "architect explorer skeptic" ;;
    esac
}

count_active_jobs() {
    if [ -f queues/active.txt ]; then
        grep -v '^#' queues/active.txt | grep -v '^$' | wc -l | tr -d ' '
    else
        echo "0"
    fi
}

# ═══════════════════════════════════════════════════════════════════════
# STATUS — Show current wave state + suggest next action
# ═══════════════════════════════════════════════════════════════════════
status_mode() {
    local wave
    wave=$(get_current_wave)
    local phase
    phase=$(detect_phase)
    local jobs
    jobs=$(count_active_jobs)

    echo "═══ Wave Status ═══"
    echo ""
    echo "  Current wave:  $wave"
    echo "  Phase:         $phase"
    echo "  Active queue:  $jobs jobs"
    echo ""

    # Check fleet
    if [ -f .claude/skills/fleet/scripts/fleet_status.sh ]; then
        local running
        running=$(bash .claude/skills/fleet/scripts/fleet_status.sh 2>/dev/null | grep -c "training" || echo "0")
        echo "  GPUs training: $running"
    fi

    # Latest results
    echo ""
    echo "── Latest Results ──"
    for f in $(ls -t results/*/summary.json results/*/summary.txt 2>/dev/null | head -5); do
        local name
        name=$(dirname "$f" | sed 's|results/||')
        local bpb="?"
        if [[ "$f" == *.json ]]; then
            bpb=$(python3 -c "
import json
d = json.load(open('$f'))
print(d.get('val_bpb') or d.get('final_quant_eval', {}).get('val_bpb', '?'))
" 2>/dev/null || echo "?")
        fi
        echo "  $name: $bpb BPB"
    done

    # Latest wave plan
    local latest_plan
    latest_plan=$(ls -t queues/wave_*_plan.md 2>/dev/null | head -1)
    if [ -n "$latest_plan" ]; then
        echo ""
        echo "── Latest Plan ──"
        local plan_status
        plan_status=$(grep "Status:" "$latest_plan" | head -1 | sed 's/.*Status: *//')
        echo "  $(basename "$latest_plan"): $plan_status"
    fi

    # Suggest next action
    echo ""
    echo "── Suggested Next Action ──"

    if [ "$jobs" -gt 0 ]; then
        # Check if experiments are actually running
        local running=0
        if [ -f .claude/skills/fleet/scripts/fleet_status.sh ]; then
            running=$(bash .claude/skills/fleet/scripts/fleet_status.sh 2>/dev/null | grep -c "training" || echo "0")
        fi

        if [ "$running" -gt 0 ]; then
            echo "  Experiments running. Use /status to monitor progress."
        else
            echo "  Queue has $jobs jobs but no GPUs training."
            echo "  → /deploy queues/active.txt"
        fi
    elif [ -n "$latest_plan" ] && echo "$plan_status" | grep -qi "pending"; then
        echo "  Plan pending approval."
        echo "  → /wave approve"
    else
        local suggested_type
        suggested_type=$(detect_debate_type)
        echo "  No active work. Ready for next wave."
        echo "  → /wave plan   (auto: $suggested_type debate)"
    fi
}

# ═══════════════════════════════════════════════════════════════════════
# PLAN — Run debate + generate wave plan
# ═══════════════════════════════════════════════════════════════════════
plan_mode() {
    local debate_type=""
    local topic=""

    # Parse options
    while [ $# -gt 0 ]; do
        case "$1" in
            type=*) debate_type="${1#type=}" ;;
            topic=*) topic="${1#topic=}" ;;
            direction|scale|pivot) debate_type="$1" ;;
            *) topic="$1" ;;
        esac
        shift
    done

    # Auto-detect debate type if not specified
    if [ -z "$debate_type" ]; then
        debate_type=$(detect_debate_type)
    fi

    # Auto-generate topic if not specified
    if [ -z "$topic" ]; then
        case "$debate_type" in
            direction)
                topic="Propose new research directions for explore experiments (500 steps, 4-8 ideas)"
                ;;
            scale)
                local latest
                latest=$(ls -t results/explore/*/summary.json results/validate/*/summary.json 2>/dev/null | head -1)
                local name="recent winners"
                [ -n "$latest" ] && name=$(dirname "$latest" | sed 's|results/||')
                topic="Scale decision for $name: validate strategy (2000-4000 steps, 2 seeds) and kill criteria"
                ;;
            pivot)
                topic="Strategic reassessment: are we saturated? What is the highest expected-value path forward?"
                ;;
        esac
    fi

    local wave_num
    wave_num=$(get_next_wave)
    local agents
    agents=$(agents_for_type "$debate_type")

    echo "═══ Wave $wave_num — $debate_type debate ═══"
    echo ""
    echo "  Topic:   $topic"
    echo "  Agents:  $agents"
    echo "  Type:    $debate_type"
    echo ""

    # Run debate with type
    bash debate/debate.sh "$topic" 3 "$debate_type"

    # Generate wave plan from synthesis
    local debate_out
    debate_out=$(ls -td debate/rounds/*/ | head -1)

    echo ""
    echo "Generating wave_${wave_num}_plan.md..."

    python3 - "$wave_num" "$debate_type" "$debate_out" "$topic" <<'PYEOF'
import sys, os, glob, datetime

wave_num = sys.argv[1]
debate_type = sys.argv[2]
debate_dir = sys.argv[3]
topic = sys.argv[4]

os.chdir(os.environ.get('REPO_ROOT', '/root/parameter-golf'))

synth_file = os.path.join(debate_dir, 'synthesis.md')
synth = ""
if os.path.exists(synth_file):
    with open(synth_file) as f:
        synth = f.read()

# Extract queue entries from synthesis
queue_lines = []
in_code_block = False
for line in synth.split('\n'):
    if '```bash' in line or '```shell' in line:
        in_code_block = True
        continue
    if in_code_block and '```' in line:
        in_code_block = False
        continue
    if in_code_block and ('run_experiment' in line or ('=' in line and not line.startswith('#'))):
        clean = line.strip()
        if clean and not clean.startswith('```'):
            queue_lines.append(clean)
    elif 'run_experiment.sh' in line:
        clean = line.strip().lstrip('#').strip()
        queue_lines.append(clean)

# Write queue file
queue_file = f'queues/wave_{wave_num}.txt'
with open(queue_file, 'w') as f:
    f.write(f"# Wave {wave_num} — {debate_type} debate\n")
    f.write(f"# Topic: {topic}\n")
    f.write(f"# Source: {debate_dir}\n")
    f.write(f"# Review and edit before approving with /wave approve\n\n")
    for line in queue_lines:
        f.write(line + '\n')

print(f"  Queue: {queue_file} ({len(queue_lines)} entries)")

# Write plan file
plan_file = f'queues/wave_{wave_num}_plan.md'
with open(plan_file, 'w') as f:
    f.write(f"""# Wave {wave_num} — {topic}

**Created:** {datetime.date.today().isoformat()}
**Debate type:** {debate_type.title()}
**Status:** PENDING

---

## Decision
{topic}

## Debate Summary
<!-- Auto-generated from {debate_dir} -->

{synth[:3000] if synth else '(synthesis not found — check debate output)'}

## Kill Criteria
- Improvement < 0.005 BPB at target step count -> drop
- Size violation (>16 MB) -> redesign or drop
- Seed divergence > 0.005 BPB -> noise, drop
- Budget exceeded -> stop

## Advancement Gate
- Explore winners (>0.01 BPB) -> Scale debate for validate plan
- Validate winners (>0.005 BPB, 2 seeds agree) -> Scale debate for full run
- Full run beats 1.2244 BPB -> submit to leaderboard

## Budget Estimate
(Estimate based on {len(queue_lines)} experiments — edit before approving)

---

*Approved by:* _________  *Date:* _________
""")

print(f"  Plan:  {plan_file}")
PYEOF

    echo ""
    echo "Review and edit:"
    echo "  queues/wave_${wave_num}_plan.md  ← plan"
    echo "  queues/wave_${wave_num}.txt      ← queue entries"
    echo ""
    echo "When ready: /wave approve"
}

# ═══════════════════════════════════════════════════════════════════════
# APPROVE — Lock plan, activate queue
# ═══════════════════════════════════════════════════════════════════════
approve_mode() {
    # Find latest pending plan
    local plan=""
    for f in $(ls -t queues/wave_*_plan.md 2>/dev/null); do
        if grep -q "PENDING" "$f" 2>/dev/null; then
            plan="$f"
            break
        fi
    done

    if [ -z "$plan" ]; then
        echo "No pending wave plan found."
        echo "Run /wave plan first."
        return 1
    fi

    local wave_num
    wave_num=$(echo "$plan" | sed 's|queues/wave_||; s|_plan.md||')
    local queue="queues/wave_${wave_num}.txt"

    if [ ! -f "$queue" ]; then
        echo "Queue file not found: $queue"
        return 1
    fi

    echo "═══ Approving Wave $wave_num ═══"
    echo ""
    echo "── Plan ──"
    head -20 "$plan"
    echo ""
    echo "── Queue ($queue) ──"
    cat "$queue"
    echo ""

    local job_count
    job_count=$(grep -v '^#' "$queue" | grep -v '^$' | wc -l | tr -d ' ')
    echo "Total: $job_count experiments"
    echo ""

    # Update plan status
    sed -i 's/Status: PENDING/Status: APPROVED/' "$plan"

    # Activate queue
    cp "$queue" queues/active.txt
    echo "  Activated: queues/active.txt ($job_count jobs)"
    echo "  Plan status: APPROVED"
    echo ""
    echo "Next: /deploy queues/active.txt"
}

# ═══════════════════════════════════════════════════════════════════════
# RESULTS — Collect, compare, draft X post, suggest next action
# ═══════════════════════════════════════════════════════════════════════
results_mode() {
    echo "═══ Wave Results ═══"
    echo ""

    # Step 1: Collect
    echo "── Step 1: Collecting results from GPUs ──"
    if [ -f .claude/skills/collect/scripts/collect.sh ]; then
        bash .claude/skills/collect/scripts/collect.sh 2>/dev/null || echo "(collect skipped — run /collect manually if needed)"
    else
        echo "(collect script not found — results may already be local)"
    fi
    echo ""

    # Step 2: Detect phase and find relevant results
    local phase
    phase=$(detect_phase)
    echo "── Step 2: Analyzing ($phase phase) ──"
    echo ""

    # Find recent results
    local results_found=0
    echo "Recent results:"
    for f in $(ls -t results/*/summary.json results/*/summary.txt 2>/dev/null | head -10); do
        local name
        name=$(dirname "$f" | sed 's|results/||')
        local bpb="?"
        local steps="?"
        if [[ "$f" == *.json ]]; then
            read -r bpb steps < <(python3 -c "
import json
d = json.load(open('$f'))
bpb = d.get('val_bpb') or d.get('final_quant_eval', {}).get('val_bpb', '?')
steps = d.get('step', d.get('total_steps', '?'))
print(f'{bpb} {steps}')
" 2>/dev/null || echo "? ?")
        fi
        printf "  %-50s %s BPB @ %s steps\n" "$name" "$bpb" "$steps"
        results_found=$((results_found + 1))
    done

    if [ "$results_found" -eq 0 ]; then
        echo "  No results found. Experiments may still be running."
        echo "  → /status to check progress"
        return
    fi

    # Step 3: Gate check
    echo ""
    echo "── Step 3: Gate Check ($phase) ──"
    case "$phase" in
        explore)
            echo "  Explore gate: any winner >0.01 BPB over baseline (1.4793 @ 500 steps)?"
            echo "  Review results above. Winners advance to validate."
            echo ""
            echo "  If winners found:  /wave plan type=scale"
            echo "  If no winners:     /wave plan type=direction  (or /wave pivot)"
            ;;
        validate)
            echo "  Validate gate: seeds agree within 0.005 BPB?"
            echo "  Review results above. Winner advances to full."
            echo ""
            echo "  If winner holds:   /wave plan type=scale  (for full run)"
            echo "  If doesn't hold:   /wave plan type=direction"
            ;;
        full)
            echo "  Full gate: beats 1.2244 BPB?"
            echo "  Review results above."
            echo ""
            echo "  If beats target:   Submit to leaderboard!"
            echo "  If doesn't beat:   /wave pivot"
            ;;
        *)
            echo "  Could not determine phase. Review results manually."
            ;;
    esac

    # Step 4: Draft X post
    echo ""
    echo "── Step 4: X Post Draft ──"
    echo ""

    case "$phase" in
        explore)
            echo "--- DRAFT (edit before posting) ---"
            echo ""
            echo "Parameter Golf — Explore Results (Wave $(get_current_wave))"
            echo ""
            echo "Quick screen: $results_found experiments at 500 steps"
            echo ""
            echo "Results:"
            for f in $(ls -t results/explore/*/summary.json 2>/dev/null | head -5); do
                local name bpb
                name=$(dirname "$f" | sed 's|results/explore/||')
                bpb=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('val_bpb','?'))" 2>/dev/null || echo "?")
                echo "  $name: $bpb BPB"
            done
            echo ""
            echo "Next: [advancing winners to 4000-step validation / pivoting to new direction]"
            echo ""
            echo "#ParameterGolf #ML"
            echo ""
            echo "--- END DRAFT ---"
            ;;
        validate)
            echo "--- DRAFT (edit before posting) ---"
            echo ""
            echo "Parameter Golf — Validation Results (Wave $(get_current_wave))"
            echo ""
            for f in $(ls -t results/validate/*/summary.json 2>/dev/null | head -4); do
                local name bpb
                name=$(dirname "$f" | sed 's|results/validate/||')
                bpb=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('val_bpb','?'))" 2>/dev/null || echo "?")
                echo "  $name: $bpb BPB"
            done
            echo ""
            echo "Seed consistency: [check above]"
            echo "Decision: [advancing to full / dropping]"
            echo ""
            echo "#ParameterGolf #ML"
            echo ""
            echo "--- END DRAFT ---"
            ;;
        full)
            echo "--- DRAFT (edit before posting) ---"
            echo ""
            echo "Parameter Golf — Full Run Complete (Wave $(get_current_wave))"
            echo ""
            echo "Config: [describe model]"
            for f in $(ls -t results/full/*/summary.json 2>/dev/null | head -2); do
                local name bpb
                name=$(dirname "$f" | sed 's|results/full/||')
                bpb=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('val_bpb','?'))" 2>/dev/null || echo "?")
                echo "  $name: $bpb BPB"
            done
            echo ""
            echo "Leaderboard: 1.2244 BPB"
            echo "Our result: [fill in]"
            echo ""
            echo "#ParameterGolf #ML"
            echo ""
            echo "--- END DRAFT ---"
            ;;
    esac

    echo ""
    echo "Use /post to refine the draft with images."

    # Step 5: Archive
    echo ""
    echo "── Step 5: Archive ──"
    local wave_num
    wave_num=$(get_current_wave)
    if [ -f "queues/wave_${wave_num}.txt" ] && [ ! -d "queues/archive/wave_${wave_num}" ]; then
        echo "  To archive: mkdir -p queues/archive/wave_${wave_num} && mv queues/wave_${wave_num}.txt queues/archive/wave_${wave_num}/"
    fi
    if [ -f queues/active.txt ]; then
        echo "  To clear active queue: rm queues/active.txt"
    fi
}

# ═══════════════════════════════════════════════════════════════════════
# PIVOT — Force a pivot debate
# ═══════════════════════════════════════════════════════════════════════
pivot_mode() {
    local topic="${1:-Strategic reassessment: are we saturated? What is the highest expected-value path forward?}"
    plan_mode "type=pivot" "topic=$topic"
}

# ─── Dispatch ─────────────────────────────────────────────────────────
case "$MODE" in
    status|"")  status_mode ;;
    plan)       plan_mode "$@" ;;
    approve)    approve_mode ;;
    results)    results_mode ;;
    pivot)      pivot_mode "$@" ;;
    *)          echo "Unknown mode: $MODE"; echo "Usage: /wave [status|plan|approve|results|pivot]"; exit 1 ;;
esac
