#!/bin/bash
# Watch training progress across all 4 GPUs with cost tracking
# Usage: watch -n 30 bash lab/watch_all_gpus.sh
# Reset budget timer: rm /tmp/gpu_watch_start

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/gpu_creds.sh"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5 -o LogLevel=ERROR"
R="/root/parameter-golf"
STATE_FILE="/tmp/gpu_watch_start"
BUDGET=40.00

# $/hr per GPU (Novita rates)
COST_LOCAL3090=0.10
COST_REMOTE3090=0.10
COST_L40S=0.28
COST_5090=0.30

# Track session start
[ ! -f "$STATE_FILE" ] && date +%s > "$STATE_FILE"
START_EPOCH=$(cat "$STATE_FILE")
NOW_EPOCH=$(date +%s)
ELAPSED_S=$((NOW_EPOCH - START_EPOCH))
ELAPSED_DISPLAY=$(printf "%dh%02dm" $((ELAPSED_S/3600)) $(( (ELAPSED_S%3600)/60 )))

# Cost math via awk (no bc dependency)
calc() { awk "BEGIN{printf \"%.4f\", $1}"; }
calci() { awk "BEGIN{printf \"%.0f\", $1}"; }

TOTAL_HOURLY=$(calc "$COST_LOCAL3090 + $COST_REMOTE3090 + $COST_L40S + $COST_5090")
ELAPSED_H=$(calc "$ELAPSED_S / 3600.0")
TOTAL_SPENT=$(calc "$ELAPSED_H * $TOTAL_HOURLY")
REMAINING=$(calc "$BUDGET - $TOTAL_SPENT")
COST_PER_MIN=$(calc "$TOTAL_HOURLY / 60.0")
HOURS_LEFT=$(calc "$REMAINING / $TOTAL_HOURLY")
PCT=$(calci "$REMAINING / $BUDGET * 100")

# Budget bar
BAR_FILL=$((PCT / 5))
[ $BAR_FILL -gt 20 ] && BAR_FILL=20
[ $BAR_FILL -lt 0 ] && BAR_FILL=0
BAR_EMPTY=$((20 - BAR_FILL))
BAR=""
for ((i=0; i<BAR_FILL; i++)); do BAR+="█"; done
for ((i=0; i<BAR_EMPTY; i++)); do BAR+="░"; done

# Remote fetch via base64 (no quoting issues)
ssh_run() {
    local port="$1" pass="$2" script="$3"
    local b64=$(echo "cd $R"$'\n'"$script" | base64 -w0)
    sshpass -p "$pass" ssh $SSH_OPTS -p "$port" root@"$HOST" "echo $b64 | base64 -d | bash" 2>/dev/null
}

# Parallel fetch into temp files
TMP_LOCAL=$(mktemp) TMP_R3090=$(mktemp) TMP_L40S=$(mktemp) TMP_5090=$(mktemp)

(   f=$(ls -t "$R"/logs/*.txt 2>/dev/null | head -1)
    [ -n "$f" ] && step=$(tail -1 "$f")
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null)
    echo "${step}|||${gpu}" > "$TMP_LOCAL"
) &

for spec in "$GPU_REMOTE3090_PORT $GPU_REMOTE3090_PASS $TMP_R3090" \
            "$GPU_L40S_PORT $GPU_L40S_PASS $TMP_L40S" \
            "$GPU_5090_PORT $GPU_5090_PASS $TMP_5090"; do
    read -r port pass tmp <<< "$spec"
    (
        step=$(ssh_run "$port" "$pass" 'f=$(ls -t logs/*.txt 2>/dev/null | head -1); [ -n "$f" ] && tail -1 "$f"')
        gpu=$(ssh_run "$port" "$pass" 'nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null')
        echo "${step}|||${gpu}" > "$tmp"
    ) &
done
wait

# Parse
get_step() { local d=$(cat "$1"); echo "${d%|||*}"; }
get_gpu()  { local d=$(cat "$1"); echo "${d#*|||}"; }

show() {
    local label="$1" cost_hr="$2" step="$3" gpu="$4"
    local cost1k=""
    case "$step" in
        step:*)
            # Extract step_avg from line like "step_avg:605.01ms"
            local avg_ms=$(echo "$step" | grep -oP 'step_avg:\K[0-9.]+' || true)
            if [ -n "$avg_ms" ]; then
                cost1k=$(awk "BEGIN{printf \"  \$%.3f/1k\", $cost_hr * $avg_ms / 3600000.0 * 1000}")
            fi
            ;;
        *) step="(idle)" ;;
    esac
    printf "%-26s | %-58s | %s\n" "$label \$${cost_hr}/hr${cost1k}" "$step" "${gpu:-offline}"
}

# Output
echo "=== GPU Fleet @ $(date '+%H:%M:%S') | Uptime: $ELAPSED_DISPLAY | Fleet: \$${TOTAL_HOURLY}/hr ==="
printf "Budget: \$%.2f / \$%.2f  [%s]  %s%%  (~%sh left)\n" "$REMAINING" "$BUDGET" "$BAR" "$PCT" "$HOURS_LEFT"
echo "--------------------------+------------------------------------------------------------+--------------------"
printf "%-26s | %-58s | %s\n" "GPU" "Progress" "Util%, Temp°C, Watts"
echo "--------------------------+------------------------------------------------------------+--------------------"
show "LOCAL 3090"  "$COST_LOCAL3090"  "$(get_step $TMP_LOCAL)" "$(get_gpu $TMP_LOCAL)"
show "REMOTE 3090" "$COST_REMOTE3090" "$(get_step $TMP_R3090)" "$(get_gpu $TMP_R3090)"
show "L40S"         "$COST_L40S"       "$(get_step $TMP_L40S)"  "$(get_gpu $TMP_L40S)"
show "RTX 5090"     "$COST_5090"       "$(get_step $TMP_5090)"  "$(get_gpu $TMP_5090)"
echo "--------------------------+------------------------------------------------------------+--------------------"
printf "Spent: \$%.2f  |  Burn: \$%.4f/min  |  Per GPU/day: local=\$2.40 remote=\$2.40 l40s=\$6.72 5090=\$7.20\n" "$TOTAL_SPENT" "$COST_PER_MIN"

rm -f "$TMP_LOCAL" "$TMP_R3090" "$TMP_L40S" "$TMP_5090"
