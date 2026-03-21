#!/usr/bin/env bash
# Generate GPU fleet cost report.
# Usage: bash cost_report.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
FLEET_SCRIPTS="$REPO_ROOT/.claude/skills/fleet/scripts"

BUDGET=40.00

# GPU hourly rates (Novita AI pricing)
# These are defaults — override in gpu_creds.sh by setting GPU_<NAME>_RATE
declare -A RATES
RATES=(
    [REMOTE3090]=0.10
    [L40S]=0.28
    [5090]=0.30
    [ARCH1]=0.10
    [ARCH2]=0.10
    [ARCH3]=0.10
    [ARCH4]=0.10
)

# Session tracking
START_FILE="/tmp/gpu_watch_start"
if [ -f "$START_FILE" ]; then
    START_TS=$(cat "$START_FILE")
    NOW_TS=$(date +%s)
    ELAPSED_S=$((NOW_TS - START_TS))
    ELAPSED_H=$(python3 -c "print(f'{$ELAPSED_S/3600:.2f}')")
else
    echo "No session tracking file found at $START_FILE"
    echo "Cost tracking starts when lab/watch_all_gpus.sh runs."
    ELAPSED_S=0
    ELAPSED_H="0.00"
fi

echo "=== GPU Fleet Cost Report ==="
echo "Session duration: ${ELAPSED_H} hours"
echo "Budget: \$${BUDGET}"
echo ""

printf "%-15s %-10s %-12s %-12s %-10s\n" "GPU" "Rate/hr" "Status" "Session \$" "Daily \$"
printf "%-15s %-10s %-12s %-12s %-10s\n" "---" "-------" "------" "---------" "------"

TOTAL_RATE=0
TOTAL_SPENT=0

"$FLEET_SCRIPTS/discover_gpus.sh" 2>/dev/null | while read -r name port pass; do
    rate=${RATES[$name]:-0.10}

    # Check if GPU is running
    status=$(bash "$FLEET_SCRIPTS/ssh_run.sh" "$port" "$pass" \
        'ps aux | grep train_gpt.py | grep -v grep > /dev/null && echo "RUNNING" || echo "IDLE"' 2>/dev/null) || status="OFFLINE"

    spent=$(python3 -c "print(f'{$rate * $ELAPSED_H:.2f}')" 2>/dev/null || echo "0.00")
    daily=$(python3 -c "print(f'{$rate * 24:.2f}')" 2>/dev/null || echo "0.00")

    printf "%-15s \$%-9s %-12s \$%-11s \$%-9s\n" "$name" "$rate" "$status" "$spent" "$daily"
done

echo ""

# Summary
python3 -c "
rates = $( python3 -c "
import re
rates = {'REMOTE3090': 0.10, 'L40S': 0.28, '5090': 0.30, 'ARCH1': 0.10, 'ARCH2': 0.10, 'ARCH3': 0.10, 'ARCH4': 0.10}
print(rates)
" )
total_rate = sum(rates.values())
elapsed_h = $ELAPSED_H
total_spent = total_rate * elapsed_h
remaining = $BUDGET - total_spent
hours_left = remaining / total_rate if total_rate > 0 else float('inf')

print(f'Fleet rate:     \${total_rate:.2f}/hr')
print(f'Total spent:    \${total_spent:.2f}')
print(f'Remaining:      \${remaining:.2f}')
print(f'Time to exhaust: {hours_left:.1f} hours ({hours_left/24:.1f} days)')
if remaining < $BUDGET * 0.2:
    print()
    print('⚠️  WARNING: Budget below 20%!')
" 2>/dev/null
