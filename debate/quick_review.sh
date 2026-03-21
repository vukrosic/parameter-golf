#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Quick Multi-Agent Review (single round, no debate)
# ═══════════════════════════════════════════════════════════════════════
# All agents review in parallel, Claude synthesizes. ~5 min total.
#
# Usage:
#   bash debate/quick_review.sh                    # default review
#   bash debate/quick_review.sh "should we try MoE at dim 448?"
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(dirname "$0")/.."
source debate/creds.sh

PROMPT="${1:-Read KNOWLEDGE.md and the latest results. What are the 3 highest-value experiments we should run next? Be specific with env var configs.}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="debate/rounds/$TIMESTAMP"
mkdir -p "$OUT_DIR"

echo "=== Quick Multi-Agent Review ==="
echo "Prompt: $PROMPT"
echo ""

# Load personas
P_ARCH=$(cat debate/personas/architect.md)
P_SKEP=$(cat debate/personas/skeptic.md)
P_EXPL=$(cat debate/personas/explorer.md)

FULL_PROMPT="$PROMPT

Read KNOWLEDGE.md for proven facts and failed approaches, then check results/explore/ and results/validate/ for recent experiments. Provide your analysis following your persona's output format."

# Parallel reviews
echo "  Launching all agents..."
claude --print -p "$P_ARCH

$FULL_PROMPT" > "$OUT_DIR/architect.md" 2>/dev/null &
PID1=$!

ANTHROPIC_BASE_URL="$NOVITA_BASE_URL" ANTHROPIC_AUTH_TOKEN="$NOVITA_KEY" \
ANTHROPIC_MODEL="$NOVITA_MODEL_MINIMAX" ANTHROPIC_SMALL_FAST_MODEL="$NOVITA_MODEL_MINIMAX" \
claude --print -p "$P_SKEP

$FULL_PROMPT" > "$OUT_DIR/skeptic.md" 2>/dev/null &
PID2=$!

ANTHROPIC_BASE_URL="$NOVITA_BASE_URL" ANTHROPIC_AUTH_TOKEN="$NOVITA_KEY" \
ANTHROPIC_MODEL="$NOVITA_MODEL_KIMI" ANTHROPIC_SMALL_FAST_MODEL="$NOVITA_MODEL_KIMI" \
claude --print -p "$P_EXPL

$FULL_PROMPT" > "$OUT_DIR/explorer.md" 2>/dev/null &
PID3=$!

wait $PID1 && echo "  ✓ Architect" || echo "  ✗ Architect"
wait $PID2 && echo "  ✓ Skeptic" || echo "  ✗ Skeptic"
wait $PID3 && echo "  ✓ Explorer" || echo "  ✗ Explorer"

# Synthesize
echo ""
echo "  Synthesizing..."

REVIEWS=""
for F in "$OUT_DIR"/{architect,skeptic,explorer}.md; do
    NAME=$(basename "$F" .md)
    if [ -s "$F" ]; then
        REVIEWS="$REVIEWS
━━━ THE $(echo "$NAME" | tr '[:lower:]' '[:upper:]') ━━━
$(cat "$F")

"
    fi
done

claude --print -p "Three AI agents with different perspectives reviewed the same question: '$PROMPT'

$REVIEWS

Synthesize into a single actionable answer:
1. Where they agree (high confidence)
2. Where they disagree (note which side has better evidence)
3. Final ranked list of experiments with exact env var configs for run_experiment.sh" \
> "$OUT_DIR/synthesis.md" 2>/dev/null

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cat "$OUT_DIR/synthesis.md"
echo ""
echo "Full outputs: $OUT_DIR/"
