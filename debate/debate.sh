#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Multi-Agent Research Debate (Mixed models via Novita)
# ═══════════════════════════════════════════════════════════════════════
# Three agents with distinct personas and models debate experiments.
#
# Usage:
#   bash debate/debate.sh                          # full 3-round debate
#   bash debate/debate.sh "MoE budget fitting"     # focused topic
#   bash debate/debate.sh "activation functions" 1  # single round only
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
source debate/creds.sh

TOPIC="${1:-Review all recent experiments and propose the next batch}"
ROUNDS="${2:-3}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="debate/rounds/$TIMESTAMP"
PROMPT_DIR="$OUT_DIR/prompts"
mkdir -p "$OUT_DIR" "$PROMPT_DIR"

# Base Novita config (per-agent model set in run_agent)
export ANTHROPIC_BASE_URL="$NOVITA_BASE_URL"
export ANTHROPIC_AUTH_TOKEN="$NOVITA_KEY"

echo "╔═══════════════════════════════════════════════╗"
echo "║         MULTI-AGENT RESEARCH DEBATE           ║"
echo "╠═══════════════════════════════════════════════╣"
echo "║ Topic: $(echo "$TOPIC" | head -c 43)"
echo "║ Rounds: $ROUNDS"
echo "║ Output: $OUT_DIR/"
echo "║                                               ║"
echo "║ Agents:                                       ║"
echo "║   Architect — $MODEL_ARCHITECT"
echo "║   Skeptic   — $MODEL_SKEPTIC"
echo "║   Explorer  — $MODEL_EXPLORER"
echo "║   Synthesis — $MODEL_SYNTHESIS"
echo "╚═══════════════════════════════════════════════╝"
echo ""

# ─── Context instructions for all agents ─────────────────────────────

# ─── Build context snapshot (inject real files, don't ask agents to read) ──

KNOWLEDGE=$(cat KNOWLEDGE.md 2>/dev/null | head -300 || echo "(KNOWLEDGE.md not found)")
QUEUE=$(cat queues/active.txt 2>/dev/null || echo "(no active queue)")
CONCLUSIONS=$(tail -80 debate/CONCLUSIONS.md 2>/dev/null || echo "(no previous debates)")

# Grab latest result summaries (up to 10)
RESULTS=""
for f in $(ls -t results/*/summary.txt results/*/summary.json 2>/dev/null | head -10); do
    RESULTS+="── $(dirname "$f" | sed 's|results/||') ──
$(head -20 "$f")

"
done
[ -z "$RESULTS" ] && RESULTS="(no results found)"

CONTEXT="## Project Context
You are working on the Parameter Golf challenge: train the best 16MB language model in <10 min on 8xH100s. Scored by val_bpb (bits per byte, lower = better). Current leaderboard: 1.2244 BPB. Our best: 1.2498 BPB.

## Your Task
$TOPIC

## KNOWLEDGE.md (source of truth)
$KNOWLEDGE

## Latest Results
$RESULTS

## Current Queue
$QUEUE

## Previous Debate Conclusions
$CONCLUSIONS

## Rules
- NEVER propose something in KNOWLEDGE.md's Failed Approaches without explaining why this time is different
- All experiments must include exact env var configs for: infra/run_experiment.sh <name> <steps>
- Respect the 16 MB int8 zlib-compressed submission limit
- Noise floor at 500 steps is ~0.003 BPB. Do not claim significance below that.
- Name experiments in snake_case"

# ─── Helper: write prompt to file, pipe to claude with model override ──

run_agent() {
    local prompt_file="$1"
    local output_file="$2"
    local model="${3:-$DEBATE_MODEL}"
    local err_file="${output_file%.md}.err"
    export ANTHROPIC_MODEL="$model"
    export ANTHROPIC_SMALL_FAST_MODEL="$model"
    cat "$prompt_file" | claude -p > "$output_file" 2>"$err_file"
}

# ═══════════════════════════════════════════════════════════════════════
# ROUND 1: Independent Reviews (parallel)
# ═══════════════════════════════════════════════════════════════════════

echo "━━━ ROUND 1: Independent Reviews ━━━"
echo ""

R1_SUFFIX="

$CONTEXT

This is Round 1 — your independent review. No other agents have spoken yet. Read the files listed above and provide your full analysis following your output format."

# Write prompt files
cat debate/personas/architect.md > "$PROMPT_DIR/r1_architect.txt"
echo "$R1_SUFFIX" >> "$PROMPT_DIR/r1_architect.txt"

cat debate/personas/skeptic.md > "$PROMPT_DIR/r1_skeptic.txt"
echo "$R1_SUFFIX" >> "$PROMPT_DIR/r1_skeptic.txt"

cat debate/personas/explorer.md > "$PROMPT_DIR/r1_explorer.txt"
echo "$R1_SUFFIX" >> "$PROMPT_DIR/r1_explorer.txt"

echo "  Launching Architect ($MODEL_ARCHITECT)..."
run_agent "$PROMPT_DIR/r1_architect.txt" "$OUT_DIR/r1_architect.md" "$MODEL_ARCHITECT" &
PID1=$!

echo "  Launching Skeptic ($MODEL_SKEPTIC)..."
run_agent "$PROMPT_DIR/r1_skeptic.txt" "$OUT_DIR/r1_skeptic.md" "$MODEL_SKEPTIC" &
PID2=$!

echo "  Launching Explorer ($MODEL_EXPLORER)..."
run_agent "$PROMPT_DIR/r1_explorer.txt" "$OUT_DIR/r1_explorer.md" "$MODEL_EXPLORER" &
PID3=$!

echo ""
echo "  Waiting for Round 1... (tail -f $OUT_DIR/r1_*.md to watch live)"
echo ""

while kill -0 $PID1 2>/dev/null || kill -0 $PID2 2>/dev/null || kill -0 $PID3 2>/dev/null; do
    ARCH_N=$(wc -l < "$OUT_DIR/r1_architect.md" 2>/dev/null || echo 0)
    SKEP_N=$(wc -l < "$OUT_DIR/r1_skeptic.md" 2>/dev/null || echo 0)
    EXPL_N=$(wc -l < "$OUT_DIR/r1_explorer.md" 2>/dev/null || echo 0)
    printf "\r  Architect: %4s lines | Skeptic: %4s lines | Explorer: %4s lines" "$ARCH_N" "$SKEP_N" "$EXPL_N"
    sleep 10
done
echo ""

wait $PID1 && echo "  ✓ Architect done ($(wc -l < "$OUT_DIR/r1_architect.md") lines)" || echo "  ✗ Architect failed (see $OUT_DIR/r1_architect.err)"
wait $PID2 && echo "  ✓ Skeptic done ($(wc -l < "$OUT_DIR/r1_skeptic.md") lines)" || echo "  ✗ Skeptic failed (see $OUT_DIR/r1_skeptic.err)"
wait $PID3 && echo "  ✓ Explorer done ($(wc -l < "$OUT_DIR/r1_explorer.md") lines)" || echo "  ✗ Explorer failed (see $OUT_DIR/r1_explorer.err)"

[ "$ROUNDS" -le 1 ] && { echo ""; echo "Single-round mode. Outputs in $OUT_DIR/"; exit 0; }

# ═══════════════════════════════════════════════════════════════════════
# ROUND 2: Cross-Examination (parallel)
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "━━━ ROUND 2: Cross-Examination ━━━"
echo ""

CROSS_INSTRUCTIONS="This is Round 2 — Cross-Examination. Below are the other agents' Round 1 reviews.

Your job:
1. Challenge: Pick the weakest claim from each other agent and explain why it is wrong or unsupported
2. Support: Pick the strongest claim from each other agent and add evidence for it
3. Revise: Update your top 3 experiment proposals based on what you have learned
4. Hill to die on: State your single biggest disagreement with the group"

# Architect sees Skeptic + Explorer
{
    cat debate/personas/architect.md
    echo ""
    echo "$CROSS_INSTRUCTIONS"
    echo ""
    echo "══════════════════════════════════════"
    echo "Review by: THE SKEPTIC"
    echo "══════════════════════════════════════"
    cat "$OUT_DIR/r1_skeptic.md" 2>/dev/null
    echo ""
    echo "══════════════════════════════════════"
    echo "Review by: THE EXPLORER"
    echo "══════════════════════════════════════"
    cat "$OUT_DIR/r1_explorer.md" 2>/dev/null
} > "$PROMPT_DIR/r2_architect.txt"

# Skeptic sees Architect + Explorer
{
    cat debate/personas/skeptic.md
    echo ""
    echo "$CROSS_INSTRUCTIONS"
    echo ""
    echo "══════════════════════════════════════"
    echo "Review by: THE ARCHITECT"
    echo "══════════════════════════════════════"
    cat "$OUT_DIR/r1_architect.md" 2>/dev/null
    echo ""
    echo "══════════════════════════════════════"
    echo "Review by: THE EXPLORER"
    echo "══════════════════════════════════════"
    cat "$OUT_DIR/r1_explorer.md" 2>/dev/null
} > "$PROMPT_DIR/r2_skeptic.txt"

# Explorer sees Architect + Skeptic
{
    cat debate/personas/explorer.md
    echo ""
    echo "$CROSS_INSTRUCTIONS"
    echo ""
    echo "══════════════════════════════════════"
    echo "Review by: THE ARCHITECT"
    echo "══════════════════════════════════════"
    cat "$OUT_DIR/r1_architect.md" 2>/dev/null
    echo ""
    echo "══════════════════════════════════════"
    echo "Review by: THE SKEPTIC"
    echo "══════════════════════════════════════"
    cat "$OUT_DIR/r1_skeptic.md" 2>/dev/null
} > "$PROMPT_DIR/r2_explorer.txt"

echo "  Architect responding ($MODEL_ARCHITECT)..."
run_agent "$PROMPT_DIR/r2_architect.txt" "$OUT_DIR/r2_architect.md" "$MODEL_ARCHITECT" &
PID1=$!

echo "  Skeptic responding ($MODEL_SKEPTIC)..."
run_agent "$PROMPT_DIR/r2_skeptic.txt" "$OUT_DIR/r2_skeptic.md" "$MODEL_SKEPTIC" &
PID2=$!

echo "  Explorer responding ($MODEL_EXPLORER)..."
run_agent "$PROMPT_DIR/r2_explorer.txt" "$OUT_DIR/r2_explorer.md" "$MODEL_EXPLORER" &
PID3=$!

echo ""
echo "  Waiting for Round 2... (tail -f $OUT_DIR/r2_*.md to watch live)"
echo ""

while kill -0 $PID1 2>/dev/null || kill -0 $PID2 2>/dev/null || kill -0 $PID3 2>/dev/null; do
    ARCH_N=$(wc -l < "$OUT_DIR/r2_architect.md" 2>/dev/null || echo 0)
    SKEP_N=$(wc -l < "$OUT_DIR/r2_skeptic.md" 2>/dev/null || echo 0)
    EXPL_N=$(wc -l < "$OUT_DIR/r2_explorer.md" 2>/dev/null || echo 0)
    printf "\r  Architect: %4s lines | Skeptic: %4s lines | Explorer: %4s lines" "$ARCH_N" "$SKEP_N" "$EXPL_N"
    sleep 10
done
echo ""

wait $PID1 && echo "  ✓ Architect R2 done" || echo "  ✗ Architect R2 failed (see $OUT_DIR/r2_architect.err)"
wait $PID2 && echo "  ✓ Skeptic R2 done" || echo "  ✗ Skeptic R2 failed (see $OUT_DIR/r2_skeptic.err)"
wait $PID3 && echo "  ✓ Explorer R2 done" || echo "  ✗ Explorer R2 failed (see $OUT_DIR/r2_explorer.err)"

[ "$ROUNDS" -le 2 ] && { echo ""; echo "Two-round mode. Outputs in $OUT_DIR/"; exit 0; }

# ═══════════════════════════════════════════════════════════════════════
# ROUND 3: Synthesis
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "━━━ ROUND 3: Synthesis ━━━"
echo ""

{
    echo "You are the Synthesis Agent. Three AI researchers — an Architect (structure), a Skeptic (rigor), and an Explorer (creativity) — debated over 2 rounds about the Parameter Golf challenge."
    echo ""
    echo "Produce the FINAL ACTIONABLE OUTPUT the human researcher will use."
    echo ""
    for ROUND in r1 r2; do
        for AGENT in architect skeptic explorer; do
            F="$OUT_DIR/${ROUND}_${AGENT}.md"
            if [ -s "$F" ]; then
                echo "╔══════════════════════════════════════╗"
                echo "  ${ROUND^^}: THE $(echo "$AGENT" | tr '[:lower:]' '[:upper:]')"
                echo "╚══════════════════════════════════════╝"
                cat "$F"
                echo ""
            fi
        done
    done
    cat <<'SYNTH_FORMAT'

Output this exact structure:

## Consensus
What all three agents agree on. High-confidence bets.

## Key Disputes
Unresolved disagreements. For each: what, who has stronger evidence, what experiment settles it.

## Experiment Queue (RANKED)
Top 5-8 experiments, ranked by expected impact / cost.

For EACH:
```bash
# <name> — <description>
# Hypothesis: <what we expect>
# Proposed by: <agent(s)>
# Risk: <what could go wrong>
ENV=val ENV2=val infra/run_experiment.sh name <steps>
```

## Do NOT Run
Rejected proposals and why.

## Strategic Assessment
One paragraph: position vs leaderboard, most promising path, biggest risk.
SYNTH_FORMAT
} > "$PROMPT_DIR/synthesis.txt"

echo "  Synthesizing ($MODEL_SYNTHESIS)..."
run_agent "$PROMPT_DIR/synthesis.txt" "$OUT_DIR/synthesis.md" "$MODEL_SYNTHESIS"
echo "  ✓ Synthesis done"

# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║              DEBATE COMPLETE                  ║"
echo "╠═══════════════════════════════════════════════╣"
for F in "$OUT_DIR"/*.md; do
    SIZE=$(wc -l < "$F")
    NAME=$(basename "$F")
    printf "║   %-30s %4d lines  ║\n" "$NAME" "$SIZE"
done
echo "╚═══════════════════════════════════════════════╝"
echo ""
echo "━━━ SYNTHESIS ━━━"
echo ""
cat "$OUT_DIR/synthesis.md"

# ═══════════════════════════════════════════════════════════════════════
# Append to persistent conclusions log
# ═══════════════════════════════════════════════════════════════════════
{
    echo ""
    echo "---"
    echo ""
    echo "## $(date '+%Y-%m-%d %H:%M') — $TOPIC"
    echo ""
    echo "**Models:** Architect=$MODEL_ARCHITECT, Skeptic=$MODEL_SKEPTIC, Explorer=$MODEL_EXPLORER, Synthesis=$MODEL_SYNTHESIS"
    echo "**Rounds:** $ROUNDS | **Full output:** $OUT_DIR/"
    echo ""
    cat "$OUT_DIR/synthesis.md"
    echo ""
} >> debate/CONCLUSIONS.md
echo ""
echo "📋 Appended to debate/CONCLUSIONS.md"
