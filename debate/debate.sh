#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Multi-Agent Research Debate
# ═══════════════════════════════════════════════════════════════════════
# Three LLMs with distinct personas independently review experiments,
# cross-examine each other, and a final synthesis extracts action items.
#
# Usage:
#   bash debate/debate.sh                          # default: full review
#   bash debate/debate.sh "MoE budget fitting"     # focused topic
#   bash debate/debate.sh "activation functions" 1  # single round (no debate)
#
# Agents:
#   Claude (Anthropic)  → The Architect (structure & parameter budgets)
#   MiniMax M2.7        → The Skeptic (rigor & statistical validity)
#   Kimi K2             → The Explorer (creative connections & novel ideas)
#
# Output: debate/rounds/<timestamp>/
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

# Save debate metadata
cat > "$OUT_DIR/meta.json" <<EOF
{
  "topic": "$TOPIC",
  "rounds": $ROUNDS,
  "timestamp": "$TIMESTAMP",
  "agents": {
    "architect": {"provider": "anthropic", "model": "claude"},
    "skeptic": {"provider": "novita", "model": "minimax/minimax-m2.7"},
    "explorer": {"provider": "novita", "model": "moonshotai/kimi-k2-instruct"}
  }
}
EOF

echo "╔═══════════════════════════════════════════════╗"
echo "║         MULTI-AGENT RESEARCH DEBATE           ║"
echo "╠═══════════════════════════════════════════════╣"
echo "║ Topic: $TOPIC"
echo "║ Rounds: $ROUNDS"
echo "║ Output: $OUT_DIR/"
echo "║                                               ║"
echo "║ Agents:                                       ║"
echo "║   Architect (Claude)  — structure & budgets   ║"
echo "║   Skeptic   (MiniMax) — rigor & evidence      ║"
echo "║   Explorer  (Kimi)    — creativity & lit       ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""

# ─── Load personas ────────────────────────────────────────────────────

PERSONA_ARCHITECT=$(cat debate/personas/architect.md)
PERSONA_SKEPTIC=$(cat debate/personas/skeptic.md)
PERSONA_EXPLORER=$(cat debate/personas/explorer.md)

# ─── Build context instructions ──────────────────────────────────────
# Each agent gets Claude Code tools, so they read files themselves.
# We tell them WHAT to read, not dump the content.

CONTEXT_INSTRUCTIONS="
## Project Context
You are working on the Parameter Golf challenge: train the best 16MB language model in <10 min on 8xH100s. Scored by val_bpb (bits per byte, lower = better). Current leaderboard: 1.2244 BPB. Our best: 1.2498 BPB.

## Your Task
$TOPIC

## Required Reading (use your file-reading tools)
1. Read KNOWLEDGE.md — this is the source of truth for proven facts and failed approaches
2. Read results in results/explore/ and results/validate/ — look at summary.txt files for BPB scores
3. Read queues/active.txt — what's currently queued
4. Skim train_gpt.py lines 39-108 — the Hyperparameters class (all tunable env vars)

## Rules
- NEVER propose something listed in KNOWLEDGE.md's 'Failed Approaches' section without acknowledging it failed and explaining why this time is different
- All experiment proposals must include exact env var configs ready for: infra/run_experiment.sh <name> <steps>
- Respect the 16 MB int8 zlib-compressed submission limit
- The noise floor at 500 steps is ~0.003 BPB. Don't claim significance below that.
- Name experiments in snake_case describing what's being tested
"

# ─── Helper: run an agent ─────────────────────────────────────────────

run_agent() {
    local name="$1"
    local persona="$2"
    local prompt="$3"
    local output_file="$4"
    local provider="${5:-anthropic}"

    case "$provider" in
        anthropic)
            claude --print -p "$persona

$prompt" > "$output_file" 2>/dev/null
            ;;
        novita-minimax)
            ANTHROPIC_BASE_URL="$NOVITA_BASE_URL" \
            ANTHROPIC_AUTH_TOKEN="$NOVITA_KEY" \
            ANTHROPIC_MODEL="$NOVITA_MODEL_MINIMAX" \
            ANTHROPIC_SMALL_FAST_MODEL="$NOVITA_MODEL_MINIMAX" \
            claude --print -p "$persona

$prompt" > "$output_file" 2>/dev/null
            ;;
        novita-kimi)
            ANTHROPIC_BASE_URL="$NOVITA_BASE_URL" \
            ANTHROPIC_AUTH_TOKEN="$NOVITA_KEY" \
            ANTHROPIC_MODEL="$NOVITA_MODEL_KIMI" \
            ANTHROPIC_SMALL_FAST_MODEL="$NOVITA_MODEL_KIMI" \
            claude --print -p "$persona

$prompt" > "$output_file" 2>/dev/null
            ;;
    esac
}

# ═══════════════════════════════════════════════════════════════════════
# ROUND 1: Independent Reviews
# ═══════════════════════════════════════════════════════════════════════

echo "━━━ ROUND 1: Independent Reviews ━━━"
echo ""

ROUND1_PROMPT="$CONTEXT_INSTRUCTIONS

This is Round 1 — your independent review. No other agents have spoken yet. Read the files listed above and provide your full analysis following your output format."

echo "  Launching Architect (Claude)..."
run_agent "architect" "$PERSONA_ARCHITECT" "$ROUND1_PROMPT" "$OUT_DIR/r1_architect.md" "anthropic" &
PID_ARCH=$!

echo "  Launching Skeptic (MiniMax M2.7)..."
run_agent "skeptic" "$PERSONA_SKEPTIC" "$ROUND1_PROMPT" "$OUT_DIR/r1_skeptic.md" "novita-minimax" &
PID_SKEP=$!

echo "  Launching Explorer (Kimi K2)..."
run_agent "explorer" "$PERSONA_EXPLORER" "$ROUND1_PROMPT" "$OUT_DIR/r1_explorer.md" "novita-kimi" &
PID_EXPL=$!

echo ""
echo "  Waiting for all agents..."
wait $PID_ARCH && echo "  ✓ Architect done ($(wc -l < "$OUT_DIR/r1_architect.md") lines)" || echo "  ✗ Architect failed"
wait $PID_SKEP && echo "  ✓ Skeptic done ($(wc -l < "$OUT_DIR/r1_skeptic.md") lines)" || echo "  ✗ Skeptic failed"
wait $PID_EXPL && echo "  ✓ Explorer done ($(wc -l < "$OUT_DIR/r1_explorer.md") lines)" || echo "  ✗ Explorer failed"

[ "$ROUNDS" -le 1 ] && { echo ""; echo "Single-round mode. Outputs in $OUT_DIR/"; exit 0; }

# ═══════════════════════════════════════════════════════════════════════
# ROUND 2: Cross-Examination
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "━━━ ROUND 2: Cross-Examination ━━━"
echo ""

build_cross_prompt() {
    local agent_name="$1"
    shift
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

    echo "This is Round 2 — Cross-Examination. You have read the other agents' Round 1 reviews below.

Your job:
1. **Challenge**: Pick the weakest claim from each other agent and explain why it's wrong or unsupported
2. **Support**: Pick the strongest claim from each other agent and add evidence for it
3. **Revise**: Update your own top 3 experiment proposals based on what you've learned. Did another agent change your mind? Say so explicitly.
4. **Disagree**: State your single biggest disagreement with the group — the hill you'd die on

$others_text"
}

# Architect sees Skeptic + Explorer
CROSS_ARCH=$(build_cross_prompt "architect" "$OUT_DIR/r1_skeptic.md" "$OUT_DIR/r1_explorer.md")
echo "  Architect responding to Skeptic & Explorer..."
run_agent "architect" "$PERSONA_ARCHITECT" "$CROSS_ARCH" "$OUT_DIR/r2_architect.md" "anthropic" &
PID_ARCH=$!

# Skeptic sees Architect + Explorer
CROSS_SKEP=$(build_cross_prompt "skeptic" "$OUT_DIR/r1_architect.md" "$OUT_DIR/r1_explorer.md")
echo "  Skeptic responding to Architect & Explorer..."
run_agent "skeptic" "$PERSONA_SKEPTIC" "$CROSS_SKEP" "$OUT_DIR/r2_skeptic.md" "novita-minimax" &
PID_SKEP=$!

# Explorer sees Architect + Skeptic
CROSS_EXPL=$(build_cross_prompt "explorer" "$OUT_DIR/r1_architect.md" "$OUT_DIR/r1_skeptic.md")
echo "  Explorer responding to Architect & Skeptic..."
run_agent "explorer" "$PERSONA_EXPLORER" "$CROSS_EXPL" "$OUT_DIR/r2_explorer.md" "novita-kimi" &
PID_EXPL=$!

echo ""
echo "  Waiting for Round 2..."
wait $PID_ARCH && echo "  ✓ Architect R2 done" || echo "  ✗ Architect R2 failed"
wait $PID_SKEP && echo "  ✓ Skeptic R2 done" || echo "  ✗ Skeptic R2 failed"
wait $PID_EXPL && echo "  ✓ Explorer R2 done" || echo "  ✗ Explorer R2 failed"

[ "$ROUNDS" -le 2 ] && { echo ""; echo "Two-round mode. Outputs in $OUT_DIR/"; exit 0; }

# ═══════════════════════════════════════════════════════════════════════
# ROUND 3: Synthesis
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "━━━ ROUND 3: Synthesis ━━━"
echo ""

# Collect all debate content
ALL_CONTENT=""
for ROUND in r1 r2; do
    for AGENT in architect skeptic explorer; do
        F="$OUT_DIR/${ROUND}_${AGENT}.md"
        if [ -s "$F" ]; then
            local_label="${ROUND^^}: THE $(echo "$AGENT" | tr '[:lower:]' '[:upper:]')"
            ALL_CONTENT="$ALL_CONTENT
╔══════════════════════════════════════╗
  $local_label
╚══════════════════════════════════════╝
$(cat "$F")
"
        fi
    done
done

SYNTH_PROMPT="You are the Synthesis Agent. Three AI researchers with different specialties — an Architect (structure), a Skeptic (rigor), and an Explorer (creativity) — debated over 2 rounds about the Parameter Golf challenge.

Your job is to produce the FINAL ACTIONABLE OUTPUT. This is what the human researcher will actually use.

Here is the complete debate:
$ALL_CONTENT

═══════════════════════════════════════
Produce this exact structure:
═══════════════════════════════════════

## Consensus
What all three agents agree on. These are high-confidence bets.

## Key Disputes
The unresolved disagreements. For each:
- What's the disagreement
- Which agent has stronger evidence
- How to resolve it (what experiment would settle it)

## Experiment Queue (RANKED)
Top 5-8 experiments, ranked by expected impact / cost ratio.

For EACH experiment:
\`\`\`bash
# <experiment_name> — <one-line description>
# Hypothesis: <what we expect>
# Proposed by: <which agent(s)>
# Risk: <what could go wrong>
# Steps: <recommended step count for this stage>
ENV_VAR1=value ENV_VAR2=value infra/run_experiment.sh experiment_name <steps>
\`\`\`

## Do NOT Run
Experiments that were proposed but rejected during debate, and why.

## Strategic Assessment
One paragraph: where are we relative to the leaderboard, what's the most promising path to closing the gap, and what's our biggest risk?"

echo "  Synthesizing debate..."
run_agent "synthesis" "" "$SYNTH_PROMPT" "$OUT_DIR/synthesis.md" "anthropic"
echo "  ✓ Synthesis done"

# ═══════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║              DEBATE COMPLETE                  ║"
echo "╠═══════════════════════════════════════════════╣"
echo "║ Files:                                        ║"
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
