#!/usr/bin/env bash
# Generate GPU fleet cost report.
# Usage: bash cost_report.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
FLEET_SCRIPTS="$REPO_ROOT/.claude/skills/fleet/scripts"

BUDGET=40.00
DEFAULT_RATE=0.10

declare -A RATES
while read -r name rate; do
    [ -n "$name" ] && [ -n "$rate" ] && RATES["$name"]="$rate"
done < <(python3 "$REPO_ROOT/infra/gpu_inventory.py" rates 2>/dev/null || true)

if [ -f /tmp/gpu_watch_start ]; then
    START_TS=$(cat /tmp/gpu_watch_start)
    NOW_TS=$(date +%s)
    ELAPSED_S=$((NOW_TS - START_TS))
else
    echo "No session tracking file found at /tmp/gpu_watch_start"
    echo "Cost tracking starts when infra/watch_all_gpus.sh runs."
    ELAPSED_S=0
fi
ELAPSED_H=$(python3 -c "print(f'{${ELAPSED_S}/3600:.2f}')")

echo "=== GPU Fleet Cost Report ==="
echo "Session duration: ${ELAPSED_H} hours"
echo "Budget: \$${BUDGET}"
echo ""

printf "%-15s %-10s %-12s %-12s %-10s\n" "GPU" "Rate/hr" "Status" "Session \$" "Daily \$"
printf "%-15s %-10s %-12s %-12s %-10s\n" "---" "-------" "------" "---------" "------"

TOTAL_RATE=0
TOTAL_SPENT=0

while read -r name port pass; do
    [ -z "$name" ] && continue

    rate="${RATES[$name]:-}"
    if [ -z "$rate" ]; then
        rate_var="GPU_${name}_RATE"
        rate="${!rate_var:-}"
    fi
    [ -z "$rate" ] && rate="$DEFAULT_RATE"

    status=$(bash "$FLEET_SCRIPTS/ssh_run.sh" "$port" "$pass" \
        'ps aux | grep train_gpt.py | grep -v grep > /dev/null && echo "RUNNING" || echo "IDLE"' 2>/dev/null) || status="OFFLINE"

    spent=$(python3 -c "rate=float('$rate'); elapsed=float('$ELAPSED_H'); print(f'{rate * elapsed:.2f}')")
    daily=$(python3 -c "rate=float('$rate'); print(f'{rate * 24:.2f}')")

    TOTAL_RATE=$(python3 -c "total=float('$TOTAL_RATE'); rate=float('$rate'); print(f'{total + rate:.4f}')")
    TOTAL_SPENT=$(python3 -c "total=float('$TOTAL_SPENT'); spent=float('$spent'); print(f'{total + spent:.2f}')")

    printf "%-15s \$%-9s %-12s \$%-11s \$%-9s\n" "$name" "$rate" "$status" "$spent" "$daily"
done < <("$FLEET_SCRIPTS/discover_gpus.sh" 2>/dev/null)

echo ""

remaining=$(python3 -c "budget=float('$BUDGET'); spent=float('$TOTAL_SPENT'); print(f'{budget - spent:.2f}')")
hours_left=$(python3 -c "total=float('$TOTAL_RATE'); remaining=float('$remaining'); print(f'{remaining / total:.1f}' if total > 0 else 'inf')")

echo "=== Summary ==="
echo "Fleet rate:     \$${TOTAL_RATE}/hr"
echo "Total spent:    \$${TOTAL_SPENT}"
echo "Remaining:      \$${remaining}"
echo "Time to exhaust: ${hours_left} hours ($(python3 -c "hours=float('$hours_left'); print(f'{hours/24:.1f}')") days)"

if python3 -c "print('1' if float('$remaining') < float('$BUDGET') * 0.2 else '0')" | grep -q '^1$'; then
    echo
    echo "⚠️  WARNING: Budget below 20%!"
fi
