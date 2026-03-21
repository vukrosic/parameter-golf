#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Multi-Agent Research Debate (MiniMax M2.7 via Novita)
# ═══════════════════════════════════════════════════════════════════════
# Three MiniMax instances with distinct personas debate experiments.
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
mkdir -p "$OUT_DIR"

# All agents use Novita
export ANTHROPIC_BASE_URL="$NOVITA_BASE_URL"
export ANTHROPIC_AUTH_TOKEN="$NOVITA_KEY"
export ANTHROPIC_MODEL="$DEBATE_MODEL"
export ANTHROPIC_SMALL_FAST_MODEL="$DEBATE_MODEL"

echo "╔═══════════════════════════════════════════════╗"
echo "║         MULTI-AGENT RESEARCH DEBATE           ║"
echo "╠═══════════════════════════════════════════════╣"
echo "║ Topic: $(echo "$TOPIC" | head -c 43)"
echo "║ Rounds: $ROUNDS"
echo "║ Model: $DEBATE_MODEL (all agents)"
echo "║ Output: $OUT_DIR/"
echo "║                                               ║"
echo "║ Agents:                                       ║"
echo "║   Architect — structure & parameter budgets   ║"
echo "║   Skeptic   — rigor & statistical evidence    ║"
echo "║   Explorer  — creativity & literature         ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""

# ─── Load personas ────────────────────────────────────────────────────

PERSONA_ARCHITECT=$(cat debate/personas/architect.md)
PERSONA_SKEPTIC=$(cat debate/personas/skeptic.md)
PERSONA_EXPLORER=$(cat debate/personas/explorer.md)

# ─── Context instructions for all agents ─────────────────────────────

CONTEXT="
## Project Context
You are working on the Parameter Golf challenge: train the best 16MB language model in <10 min on 8xH100s. Scored by val_bpb (bits per byte, lower = better). Current leaderboard: 1.2244 BPB. Our best: 1.2498 BPB.

## Your Task
$TOPIC

## Required Reading (use your file-reading tools)
1. Read KNOWLEDGE.md — source of truth for proven facts and failed approaches
2. Read results in results/explore/ and results/validate/ — look at summary.txt files
3. Read queues/active.txt — what's currently queued
4. Skim train_gpt.py lines 39-108 — the Hyperparameters class (all tunable env vars)

## Rules
- NEVER propose something in KNOWLEDGE.md's 'Failed Approaches' without explaining why this time is different
- All experiments must include exact env var configs for: infra/run_experiment.sh <name> <steps>
- Respect the 16 MB int8 zlib-compressed submission limit
- Noise floor at 500 steps is ~0.003 BPB. Don't claim significance below that.
- Name experiments in snake_case
"

# ═══════════════════════════════════════════════════════════════════════
# ROUND 1: Independent Reviews (parallel)
# ═══════════════════════════════════════════════════════════════════════

echo "━━━ ROUND 1: Independent Reviews ━━━"
echo ""

R1_PROMPT="$CONTEXT

This is Round 1 — your independent review. No other agents have spoken yet. Read the files listed above and provide your full analysis following your output format."

# Log stderr to files so we can debug failures
echo "  Launching Architect..."
claude --print -p "$PERSONA_ARCHITECT

$R1_PROMPT" > "$OUT_DIR/r1_architect.md" 2>"$OUT_DIR/r1_architect.err" &
PID1=$!

echo "  Launching Skeptic..."
claude --print -p "$PERSONA_SKEPTIC

$R1_PROMPT" > "$OUT_DIR/r1_skeptic.md" 2>"$OUT_DIR/r1_skeptic.err" &
PID2=$!

echo "  Launching Explorer..."
claude --print -p "$PERSONA_EXPLORER

$R1_PROMPT" > "$OUT_DIR/r1_explorer.md" 2>"$OUT_DIR/r1_explorer.err" &
PID3=$!

echo ""
echo "  Waiting for Round 1... (tail -f $OUT_DIR/r1_*.md to watch live)"
echo ""

# Live progress monitor: print line counts every 10s
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

build_cross_prompt() {
    local others_text=""
    for other_file in "$@"; do
        local other_name
        other_name=$(basename "$other_file" .md | sed 's/r1_//')
        if [ -s "$other_file" ]; then
            others_text="$others_text
══════════════════════════════════════
Review by: THE $(echo "$other_name" | tr '[:lower:]' '[:upper:]')
══════════════════════════════════════
$(cat "$other_file")
"
        fi
    done

    echo "This is Round 2 — Cross-Examination. Below are the other agents' Round 1 reviews.

Your job:
1. **Challenge**: Pick the weakest claim from each other agent and explain why it's wrong or unsupported
2. **Support**: Pick the strongest claim from each other agent and add evidence for it
3. **Revise**: Update your top 3 experiment proposals based on what you've learned
4. **Hill to die on**: State your single biggest disagreement with the group

$others_text"
}

CROSS_ARCH=$(build_cross_prompt "$OUT_DIR/r1_skeptic.md" "$OUT_DIR/r1_explorer.md")
echo "  Architect responding..."
claude --print -p "$PERSONA_ARCHITECT

$CROSS_ARCH" > "$OUT_DIR/r2_architect.md" 2>/dev/null &
PID1=$!

CROSS_SKEP=$(build_cross_prompt "$OUT_DIR/r1_architect.md" "$OUT_DIR/r1_explorer.md")
echo "  Skeptic responding..."
claude --print -p "$PERSONA_SKEPTIC

$CROSS_SKEP" > "$OUT_DIR/r2_skeptic.md" 2>/dev/null &
PID2=$!

CROSS_EXPL=$(build_cross_prompt "$OUT_DIR/r1_architect.md" "$OUT_DIR/r1_skeptic.md")
echo "  Explorer responding..."
claude --print -p "$PERSONA_EXPLORER

$CROSS_EXPL" > "$OUT_DIR/r2_explorer.md" 2>/dev/null &
PID3=$!

echo ""
echo "  Waiting for Round 2..."
wait $PID1 && echo "  ✓ Architect R2 done" || echo "  ✗ Architect R2 failed"
wait $PID2 && echo "  ✓ Skeptic R2 done" || echo "  ✗ Skeptic R2 failed"
wait $PID3 && echo "  ✓ Explorer R2 done" || echo "  ✗ Explorer R2 failed"

[ "$ROUNDS" -le 2 ] && { echo ""; echo "Two-round mode. Outputs in $OUT_DIR/"; exit 0; }

# ═══════════════════════════════════════════════════════════════════════
# ROUND 3: Synthesis
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "━━━ ROUND 3: Synthesis ━━━"
echo ""

ALL_CONTENT=""
for ROUND in r1 r2; do
    for AGENT in architect skeptic explorer; do
        F="$OUT_DIR/${ROUND}_${AGENT}.md"
        if [ -s "$F" ]; then
            ALL_CONTENT="$ALL_CONTENT
╔══════════════════════════════════════╗
  ${ROUND^^}: THE $(echo "$AGENT" | tr '[:lower:]' '[:upper:]')
╚══════════════════════════════════════╝
$(cat "$F")
"
        fi
    done
done

SYNTH_PROMPT="You are the Synthesis Agent. Three AI researchers — an Architect (structure), a Skeptic (rigor), and an Explorer (creativity) — debated over 2 rounds about the Parameter Golf challenge.

Produce the FINAL ACTIONABLE OUTPUT the human researcher will use.

$ALL_CONTENT

═══════════════════════════════════════
Output this exact structure:
═══════════════════════════════════════

## Consensus
What all three agents agree on. High-confidence bets.

## Key Disputes
Unresolved disagreements. For each: what, who has stronger evidence, what experiment settles it.

## Experiment Queue (RANKED)
Top 5-8 experiments, ranked by expected impact / cost.

For EACH:
\`\`\`bash
# <name> — <description>
# Hypothesis: <what we expect>
# Proposed by: <agent(s)>
# Risk: <what could go wrong>
ENV=val ENV2=val infra/run_experiment.sh name <steps>
\`\`\`

## Do NOT Run
Rejected proposals and why.

## Strategic Assessment
One paragraph: position vs leaderboard, most promising path, biggest risk."

echo "  Synthesizing..."
claude --print -p "$SYNTH_PROMPT" > "$OUT_DIR/synthesis.md" 2>/dev/null
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
