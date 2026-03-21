#!/bin/bash
# Watch training progress across all 4 GPUs with cost tracking.
# Usage: watch -c -n 30 bash lab/watch_all_gpus.sh
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
ELAPSED_DISPLAY=$(printf "%dh%02dm" $((ELAPSED_S / 3600)) $(((ELAPSED_S % 3600) / 60)))

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

FULL_TABLE_WIDTH=118
FULL_LAYOUT_MIN=120

get_term_cols() {
    local cols="${COLUMNS:-}"

    if [[ "$cols" =~ ^[0-9]+$ ]] && ((cols > 0)); then
        printf '%s' "$cols"
        return
    fi

    if [[ -t 1 ]]; then
        cols=$(tput cols 2>/dev/null)
        if [[ "$cols" =~ ^[0-9]+$ ]] && ((cols > 0)); then
            printf '%s' "$cols"
            return
        fi
    fi

    printf '120'
}

TERM_COLS=$(get_term_cols)
USE_FULL_LAYOUT=0
((TERM_COLS >= FULL_LAYOUT_MIN)) && USE_FULL_LAYOUT=1

if [[ -z "${NO_COLOR:-}" && -t 1 && "${TERM:-}" != "dumb" ]]; then
    C_RESET=$'\033[0m'
    C_BOLD=$'\033[1m'
    C_DIM=$'\033[2m'
    C_RED=$'\033[31m'
    C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'
    C_BLUE=$'\033[34m'
    C_MAGENTA=$'\033[35m'
    C_CYAN=$'\033[36m'
else
    C_RESET=""
    C_BOLD=""
    C_DIM=""
    C_RED=""
    C_GREEN=""
    C_YELLOW=""
    C_BLUE=""
    C_MAGENTA=""
    C_CYAN=""
fi

repeat_char() {
    local char="$1" count="$2" out=""
    local i
    for ((i = 0; i < count; i++)); do
        out+="$char"
    done
    printf '%s' "$out"
}

trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

clamp_percent() {
    local n="$1"
    ((n > 100)) && n=100
    ((n < 0)) && n=0
    printf '%s' "$n"
}

fmt_ms() {
    awk -v ms="$1" 'BEGIN{
        if (ms == "" || ms == "-") { printf "-"; exit }
        if (ms >= 100) printf "%.0fms", ms
        else if (ms >= 10) printf "%.1fms", ms
        else printf "%.2fms", ms
    }'
}

fmt_hours() {
    awk -v h="$1" 'BEGIN{
        if (h < 0) h = 0
        printf "%.1f", h
    }'
}

fmt_watts() {
    awk -v w="$1" 'BEGIN{
        if (w == "" || w == "[Not Supported]") { printf "-"; exit }
        printf "%.0fW", w
    }'
}

budget_color() {
    local pct="$1"
    if ((pct >= 50)); then
        printf '%s' "$C_GREEN"
    elif ((pct >= 20)); then
        printf '%s' "$C_YELLOW"
    else
        printf '%s' "$C_RED"
    fi
}

state_color() {
    case "$1" in
        train) printf '%s' "$C_GREEN" ;;
        val) printf '%s' "$C_CYAN" ;;
        warmup) printf '%s' "$C_YELLOW" ;;
        error) printf '%s' "$C_RED" ;;
        offline) printf '%s' "$C_RED" ;;
        idle) printf '%s' "$C_DIM" ;;
        *) printf '%s' "$C_BLUE" ;;
    esac
}

gpu_label_color() {
    case "$1" in
        "LOCAL 3090") printf '%s' "$C_BLUE" ;;
        "REMOTE 3090") printf '%s' "$C_MAGENTA" ;;
        "L40S") printf '%s' "$C_CYAN" ;;
        "RTX 5090") printf '%s' "$C_YELLOW" ;;
        *) printf '%s' "$C_BOLD" ;;
    esac
}

util_color() {
    local util="$1"
    if [[ ! "$util" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        printf '%s' "$C_DIM"
        return
    fi

    awk -v util="$util" -v green="$C_GREEN" -v yellow="$C_YELLOW" -v red="$C_RED" -v dim="$C_DIM" '
        BEGIN {
            if (util >= 90) printf "%s", green
            else if (util >= 50) printf "%s", yellow
            else if (util > 0) printf "%s", dim
            else printf "%s", red
        }'
}

temp_color() {
    local temp="$1"
    if [[ ! "$temp" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        printf '%s' "$C_DIM"
        return
    fi

    awk -v temp="$temp" -v green="$C_GREEN" -v yellow="$C_YELLOW" -v red="$C_RED" '
        BEGIN {
            if (temp >= 80) printf "%s", red
            else if (temp >= 70) printf "%s", yellow
            else printf "%s", green
        }'
}

power_color() {
    if [[ "$1" == "-" ]]; then
        printf '%s' "$C_DIM"
    else
        printf '%s' "$C_CYAN"
    fi
}

build_bar() {
    local pct fill empty out=""
    pct=$(clamp_percent "$1")
    fill=$((pct / 5))
    empty=$((20 - fill))

    local i
    for ((i = 0; i < fill; i++)); do
        out+="█"
    done
    for ((i = 0; i < empty; i++)); do
        out+="░"
    done
    printf '%s' "$out"
}

table_rule() {
    printf '%s\n' "+--------------+----------+-------------+------------+------------+----------+--------+--------+---------+----------+"
}

summary_rule() {
    local width="$1"
    printf '%s\n' "$(repeat_char "=" "$width")"
}

extract_step() {
    local data
    data=$(<"$1")
    printf '%s' "${data%|||*}"
}

extract_gpu() {
    local data
    data=$(<"$1")
    printf '%s' "${data#*|||}"
}

parse_progress() {
    local step_line="$1" cost_hr="$2"
    local avg_ms

    P_STATE="idle"
    P_STEP="-"
    P_LOSS="-"
    P_AVG="-"
    P_COST1K="-"

    [[ -z "$step_line" ]] && return

    if [[ "$step_line" =~ ^warmup_step:([0-9]+/[0-9]+) ]]; then
        P_STATE="warmup"
        P_STEP="${BASH_REMATCH[1]}"
    elif [[ "$step_line" =~ ^step:([0-9]+/[0-9]+) ]]; then
        P_STEP="${BASH_REMATCH[1]}"

        if [[ "$step_line" =~ val_loss:([0-9.]+) ]]; then
            P_STATE="val"
            P_LOSS="${BASH_REMATCH[1]}"
        elif [[ "$step_line" =~ train_loss:([0-9.]+) ]]; then
            P_STATE="train"
            P_LOSS="${BASH_REMATCH[1]}"
        else
            P_STATE="step"
        fi
    fi

    if [[ "$step_line" =~ step_avg:([0-9.]+)ms ]]; then
        avg_ms="${BASH_REMATCH[1]}"
        P_AVG=$(fmt_ms "$avg_ms")
        P_COST1K=$(awk -v cost="$cost_hr" -v ms="$avg_ms" 'BEGIN{printf "$%.3f", cost * ms / 3600.0}')
    fi
}

parse_gpu_stats() {
    local gpu_line="$1"
    local first_line util temp power

    G_ONLINE=0
    G_ERROR=0
    G_UTIL_RAW=""
    G_TEMP_RAW=""
    G_UTIL="-"
    G_TEMP="-"
    G_POWER="-"

    [[ -z "$gpu_line" ]] && return

    first_line="${gpu_line%%$'\n'*}"
    IFS=',' read -r util temp power _ <<< "$first_line"
    util=$(trim "$util")
    temp=$(trim "$temp")
    power=$(trim "$power")

    [[ -z "$util" ]] && return
    if [[ ! "$util" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        G_ERROR=1
        return
    fi

    G_ONLINE=1
    G_UTIL_RAW="$util"
    [[ "$temp" =~ ^[0-9]+([.][0-9]+)?$ ]] && G_TEMP_RAW="$temp"
    G_UTIL="${util}%"
    [[ -n "$G_TEMP_RAW" ]] && G_TEMP="${temp}C"
    G_POWER=$(fmt_watts "$power")
}

render_row() {
    local label="$1" cost_hr="$2" step_line="$3" gpu_line="$4"
    local label_color state state_label state_fg util_fg temp_fg power_fg

    parse_progress "$step_line" "$cost_hr"
    parse_gpu_stats "$gpu_line"

    if ((G_ERROR == 1)); then
        state="error"
        G_UTIL="ERR"
        G_TEMP="-"
        G_POWER="-"
    elif ((G_ONLINE == 0)); then
        state="offline"
        P_STEP="-"
        P_LOSS="-"
        P_AVG="-"
        P_COST1K="-"
        G_UTIL="-"
        G_TEMP="-"
        G_POWER="-"
    else
        state="$P_STATE"
    fi

    case "$state" in
        train) state_label="TRAIN" ;;
        val) state_label="VAL" ;;
        warmup) state_label="WARMUP" ;;
        error) state_label="ERROR" ;;
        offline) state_label="OFFLINE" ;;
        idle) state_label="IDLE" ;;
        *) state_label="STEP" ;;
    esac

    label_color=$(gpu_label_color "$label")
    state_fg=$(state_color "$state")
    if [[ "$state" == "error" ]]; then
        util_fg="$C_RED"
        temp_fg="$C_RED"
        power_fg="$C_RED"
    else
        util_fg=$(util_color "$G_UTIL_RAW")
        temp_fg=$(temp_color "$G_TEMP_RAW")
        power_fg=$(power_color "$G_POWER")
    fi

    printf "| %b%-12.12s%b | %b%-8.8s%b | %-11.11s | %-10.10s | %-10.10s | %-8.8s | %b%-6.6s%b | %b%-6.6s%b | %b%-7.7s%b | %-8.8s |\n" \
        "$label_color" "$label" "$C_RESET" \
        "$state_fg" "$state_label" "$C_RESET" \
        "$P_STEP" "$P_LOSS" "$P_AVG" "$P_COST1K" \
        "$util_fg" "$G_UTIL" "$C_RESET" \
        "$temp_fg" "$G_TEMP" "$C_RESET" \
        "$power_fg" "$G_POWER" "$C_RESET" \
        "\$$cost_hr"
}

render_row_compact() {
    local label="$1" cost_hr="$2" step_line="$3" gpu_line="$4"
    local label_color state state_label state_fg util_fg temp_fg power_fg

    parse_progress "$step_line" "$cost_hr"
    parse_gpu_stats "$gpu_line"

    if ((G_ERROR == 1)); then
        state="error"
        G_UTIL="ERR"
        G_TEMP="-"
        G_POWER="-"
    elif ((G_ONLINE == 0)); then
        state="offline"
        P_STEP="-"
        P_LOSS="-"
        P_AVG="-"
        P_COST1K="-"
        G_UTIL="-"
        G_TEMP="-"
        G_POWER="-"
    else
        state="$P_STATE"
    fi

    case "$state" in
        train) state_label="TRAIN" ;;
        val) state_label="VAL" ;;
        warmup) state_label="WARMUP" ;;
        error) state_label="ERROR" ;;
        offline) state_label="OFFLINE" ;;
        idle) state_label="IDLE" ;;
        *) state_label="STEP" ;;
    esac

    label_color=$(gpu_label_color "$label")
    state_fg=$(state_color "$state")
    if [[ "$state" == "error" ]]; then
        util_fg="$C_RED"
        temp_fg="$C_RED"
        power_fg="$C_RED"
    else
        util_fg=$(util_color "$G_UTIL_RAW")
        temp_fg=$(temp_color "$G_TEMP_RAW")
        power_fg=$(power_color "$G_POWER")
    fi

    printf "%b%-12.12s%b  %b%-7.7s%b  step %-11.11s loss %-8.8s avg %-8.8s\n" \
        "$label_color" "$label" "$C_RESET" \
        "$state_fg" "$state_label" "$C_RESET" \
        "$P_STEP" "$P_LOSS" "$P_AVG"
    printf "  util %b%-6.6s%b  temp %b%-6.6s%b  power %b%-7.7s%b  \$/1k %-8.8s  cost \$%s/hr\n" \
        "$util_fg" "$G_UTIL" "$C_RESET" \
        "$temp_fg" "$G_TEMP" "$C_RESET" \
        "$power_fg" "$G_POWER" "$C_RESET" \
        "$P_COST1K" "$cost_hr"
}

# Remote fetch via base64 (no quoting issues)
ssh_run() {
    local port="$1" pass="$2" script="$3"
    local b64
    b64=$(printf 'cd %s\n%s' "$R" "$script" | base64 -w0)
    sshpass -p "$pass" ssh $SSH_OPTS -p "$port" root@"$HOST" "echo $b64 | base64 -d | bash" 2>/dev/null
}

# Parallel fetch into temp files
TMP_LOCAL=$(mktemp)
TMP_R3090=$(mktemp)
TMP_L40S=$(mktemp)
TMP_5090=$(mktemp)

cleanup() {
    rm -f "$TMP_LOCAL" "$TMP_R3090" "$TMP_L40S" "$TMP_5090"
}

trap cleanup EXIT

(
    f=$(ls -t "$R"/logs/*.txt 2>/dev/null | head -1)
    [ -n "$f" ] && step=$(tail -1 "$f")
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null)
    printf '%s|||%s\n' "$step" "$gpu" > "$TMP_LOCAL"
) &

for spec in \
    "$GPU_REMOTE3090_PORT $GPU_REMOTE3090_PASS $TMP_R3090" \
    "$GPU_L40S_PORT $GPU_L40S_PASS $TMP_L40S" \
    "$GPU_5090_PORT $GPU_5090_PASS $TMP_5090"; do
    read -r port pass tmp <<< "$spec"
    (
        step=$(ssh_run "$port" "$pass" 'f=$(ls -t logs/*.txt 2>/dev/null | head -1); [ -n "$f" ] && tail -1 "$f"')
        gpu=$(ssh_run "$port" "$pass" 'nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null')
        printf '%s|||%s\n' "$step" "$gpu" > "$tmp"
    ) &
done
wait

BAR=$(build_bar "$PCT")
BUDGET_FG=$(budget_color "$PCT")
HOURS_LEFT_DISPLAY=$(fmt_hours "$HOURS_LEFT")
LOCAL_DAY=$(calc "$COST_LOCAL3090 * 24")
REMOTE_DAY=$(calc "$COST_REMOTE3090 * 24")
L40S_DAY=$(calc "$COST_L40S * 24")
RTX5090_DAY=$(calc "$COST_5090 * 24")
SUMMARY_WIDTH=$FULL_TABLE_WIDTH
((TERM_COLS > 0 && TERM_COLS < SUMMARY_WIDTH)) && SUMMARY_WIDTH=$TERM_COLS

if ((USE_FULL_LAYOUT == 1)); then
    SUMMARY_WIDTH=$FULL_TABLE_WIDTH
    summary_rule "$SUMMARY_WIDTH"
    printf "%bGPU Fleet%b  %s UTC  |  uptime %s  |  fleet \$%.2f/hr\n" \
        "$C_BOLD$C_CYAN" "$C_RESET" "$(date '+%Y-%m-%d %H:%M:%S')" "$ELAPSED_DISPLAY" "$TOTAL_HOURLY"
    printf "Budget left %b\$%.2f%b / \$%.2f  |  spent \$%.2f  |  burn \$%.4f/min  |  est left %sh\n" \
        "$BUDGET_FG" "$REMAINING" "$C_RESET" "$BUDGET" "$TOTAL_SPENT" "$COST_PER_MIN" "$HOURS_LEFT_DISPLAY"
    printf "Remaining   %b[%s]%b  %s%%\n" "$BUDGET_FG" "$BAR" "$C_RESET" "$PCT"
    summary_rule "$SUMMARY_WIDTH"

    table_rule
    printf "| %-12s | %-8s | %-11s | %-10s | %-10s | %-8s | %-6s | %-6s | %-7s | %-8s |\n" \
        "GPU" "State" "Step" "Loss" "Avg step" "\$/1k" "Util" "Temp" "Power" "\$/hr"
    table_rule
    render_row "LOCAL 3090" "$COST_LOCAL3090" "$(extract_step "$TMP_LOCAL")" "$(extract_gpu "$TMP_LOCAL")"
    render_row "REMOTE 3090" "$COST_REMOTE3090" "$(extract_step "$TMP_R3090")" "$(extract_gpu "$TMP_R3090")"
    render_row "L40S" "$COST_L40S" "$(extract_step "$TMP_L40S")" "$(extract_gpu "$TMP_L40S")"
    render_row "RTX 5090" "$COST_5090" "$(extract_step "$TMP_5090")" "$(extract_gpu "$TMP_5090")"
    table_rule
else
    summary_rule "$SUMMARY_WIDTH"
    printf "%bGPU Fleet%b  %s UTC\n" "$C_BOLD$C_CYAN" "$C_RESET" "$(date '+%Y-%m-%d %H:%M:%S')"
    printf "uptime %s  |  fleet \$%.2f/hr  |  burn \$%.4f/min\n" "$ELAPSED_DISPLAY" "$TOTAL_HOURLY" "$COST_PER_MIN"
    printf "budget %b\$%.2f%b / \$%.2f  |  spent \$%.2f  |  left %sh (%s%%)\n" \
        "$BUDGET_FG" "$REMAINING" "$C_RESET" "$BUDGET" "$TOTAL_SPENT" "$HOURS_LEFT_DISPLAY" "$PCT"
    printf "%b[%s]%b\n" "$BUDGET_FG" "$BAR" "$C_RESET"
    summary_rule "$SUMMARY_WIDTH"

    render_row_compact "LOCAL 3090" "$COST_LOCAL3090" "$(extract_step "$TMP_LOCAL")" "$(extract_gpu "$TMP_LOCAL")"
    printf '\n'
    render_row_compact "REMOTE 3090" "$COST_REMOTE3090" "$(extract_step "$TMP_R3090")" "$(extract_gpu "$TMP_R3090")"
    printf '\n'
    render_row_compact "L40S" "$COST_L40S" "$(extract_step "$TMP_L40S")" "$(extract_gpu "$TMP_L40S")"
    printf '\n'
    render_row_compact "RTX 5090" "$COST_5090" "$(extract_step "$TMP_5090")" "$(extract_gpu "$TMP_5090")"
fi

printf '\n'
printf "Per-day cost  local \$%.2f  |  remote \$%.2f  |  l40s \$%.2f  |  5090 \$%.2f\n" \
    "$LOCAL_DAY" "$REMOTE_DAY" "$L40S_DAY" "$RTX5090_DAY"
