#!/bin/bash
# Quick single-round review — all 3 personas in parallel, then synthesis
# Usage: bash debate/quick_review.sh ["optional prompt"]

set -euo pipefail
cd "$(dirname "$0")/.."
source debate/creds.sh

export ANTHROPIC_BASE_URL="$NOVITA_BASE_URL"
export ANTHROPIC_AUTH_TOKEN="$NOVITA_KEY"
export ANTHROPIC_MODEL="$DEBATE_MODEL"
export ANTHROPIC_SMALL_FAST_MODEL="$DEBATE_MODEL"

PROMPT="${1:-Read KNOWLEDGE.md and the latest results. What are the 3 highest-value experiments to run next? Be specific with env var configs.}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="debate/rounds/$TIMESTAMP"
mkdir -p "$OUT_DIR"

echo "=== Quick Review ($DEBATE_MODEL) ==="
echo ""

TASK="$PROMPT

Read KNOWLEDGE.md for proven facts and failed approaches, then check results/explore/ and results/validate/ for recent experiments. Follow your persona's output format."

echo "  Launching 3 agents..."
claude --print -p "$(cat debate/personas/architect.md)

$TASK" > "$OUT_DIR/architect.md" 2>/dev/null &
PID1=$!

claude --print -p "$(cat debate/personas/skeptic.md)

$TASK" > "$OUT_DIR/skeptic.md" 2>/dev/null &
PID2=$!

claude --print -p "$(cat debate/personas/explorer.md)

$TASK" > "$OUT_DIR/explorer.md" 2>/dev/null &
PID3=$!

wait $PID1 && echo "  ✓ Architect" || echo "  ✗ Architect"
wait $PID2 && echo "  ✓ Skeptic" || echo "  ✗ Skeptic"
wait $PID3 && echo "  ✓ Explorer" || echo "  ✗ Explorer"

echo "  Synthesizing..."
REVIEWS=""
for F in "$OUT_DIR"/{architect,skeptic,explorer}.md; do
    NAME=$(basename "$F" .md)
    [ -s "$F" ] && REVIEWS="$REVIEWS
━━━ THE $(echo "$NAME" | tr '[:lower:]' '[:upper:]') ━━━
$(cat "$F")
"
done

claude --print -p "Three agents reviewed: '$PROMPT'

$REVIEWS

Synthesize: agreements, disagreements, final ranked experiment list with exact env var configs." \
> "$OUT_DIR/synthesis.md" 2>/dev/null

echo ""
cat "$OUT_DIR/synthesis.md"
echo ""
echo "Full outputs: $OUT_DIR/"
