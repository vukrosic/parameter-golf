#!/bin/bash
# Watch training progress across all GPUs with cost tracking.
# Auto-detects GPUs from gpu_creds.sh — add new GPUs there, they appear here automatically.
# Shows git status (modified files + unpushed commits) per GPU.
# Usage: watch -c -n 30 bash lab/watch_all_gpus.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/gpu_creds.sh"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5 -o LogLevel=ERROR"
R="/root/parameter-golf"
STATE_FILE="/tmp/gpu_watch_start"
BUDGET=40.00

[ ! -f "$STATE_FILE" ] && date +%s > "$STATE_FILE"
START_EPOCH=$(cat "$STATE_FILE")
NOW_EPOCH=$(date +%s)
ELAPSED_S=$((NOW_EPOCH - START_EPOCH))
ELAPSED_DISPLAY=$(printf "%dh%02dm" $((ELAPSED_S / 3600)) $(((ELAPSED_S % 3600) / 60)))

calc()  { awk "BEGIN{printf \"%.4f\", $1}"; }
calci() { awk "BEGIN{printf \"%.0f\", $1}"; }

# ---- Auto-detect GPUs from gpu_creds.sh ----
declare -a G_NAMES G_LABELS G_PORTS G_PASSES G_COSTS

while IFS='=' read -r key val; do
    [[ "$key" =~ ^[[:space:]]*# ]] && continue
    [[ "$key" =~ ^GPU_(.+)_PORT$ ]] || continue
    gname="${BASH_REMATCH[1]}"
    port="${val//[\"\' ]/}"
    pvar="GPU_${gname}_PASS"
    pass="${!pvar//[\"\']/}"
    [[ -z "$port" || -z "$pass" ]] && continue
    case "$gname" in
        *3090*|*3090) label="REMOTE 3090" ; cost="0.10" ;;
        *L40S*)       label="L40S"        ; cost="0.28" ;;
        *5090*)       label="RTX 5090"    ; cost="0.30" ;;
        *)            label="$gname"      ; cost="0.30" ;;
    esac
    G_NAMES+=("$gname"); G_LABELS+=("$label"); G_PORTS+=("$port")
    G_PASSES+=("$pass"); G_COSTS+=("$cost")
done < "$SCRIPT_DIR/gpu_creds.sh"

# ---- Fleet totals ----
TOTAL_HOURLY=0
for c in "${G_COSTS[@]}"; do TOTAL_HOURLY=$(calc "$TOTAL_HOURLY + $c"); done

ELAPSED_H=$(calc "$ELAPSED_S / 3600.0")
TOTAL_SPENT=$(calc "$ELAPSED_H * $TOTAL_HOURLY")
REMAINING=$(calc "$BUDGET - $TOTAL_SPENT")
COST_PER_MIN=$(calc "$TOTAL_HOURLY / 60.0")
HOURS_LEFT=$(calc "$REMAINING / $TOTAL_HOURLY")
PCT=$(calci "$REMAINING / $BUDGET * 100")

FULL_TABLE_WIDTH=130
FULL_LAYOUT_MIN=130

get_term_cols() {
    local cols="${COLUMNS:-}"
    if [[ "$cols" =~ ^[0-9]+$ ]] && ((cols > 0)); then printf '%s' "$cols"; return; fi
    if [[ -t 1 ]]; then
        cols=$(tput cols 2>/dev/null)
        if [[ "$cols" =~ ^[0-9]+$ ]] && ((cols > 0)); then printf '%s' "$cols"; return; fi
    fi
    printf '130'
}

TERM_COLS=$(get_term_cols)
USE_FULL_LAYOUT=0
((TERM_COLS >= FULL_LAYOUT_MIN)) && USE_FULL_LAYOUT=1

if [[ -z "${NO_COLOR:-}" && -t 1 && "${TERM:-}" != "dumb" ]]; then
    C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
    C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'
    C_BLUE=$'\033[34m'; C_MAGENTA=$'\033[35m'; C_CYAN=$'\033[36m'
else
    C_RESET=""; C_BOLD=""; C_DIM=""
    C_RED=""; C_GREEN=""; C_YELLOW=""
    C_BLUE=""; C_MAGENTA=""; C_CYAN=""
fi

repeat_char() {
    local char="$1" count="$2" out="" i
    for ((i = 0; i < count; i++)); do out+="$char"; done
    printf '%s' "$out"
}

trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"; s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

clamp_percent() { local n="$1"; ((n>100)) && n=100; ((n<0)) && n=0; printf '%s' "$n"; }

fmt_ms() {
    awk -v ms="$1" 'BEGIN{
        if (ms==""||ms=="-"){printf "-";exit}
        if (ms>=100) printf "%.0fms",ms
        else if (ms>=10) printf "%.1fms",ms
        else printf "%.2fms",ms
    }'
}

fmt_hours() { awk -v h="$1" 'BEGIN{if(h<0)h=0; printf "%.1f",h}'; }

fmt_watts() {
    awk -v w="$1" 'BEGIN{
        if(w==""||w=="[Not Supported]"){printf "-";exit}
        printf "%.0fW",w
    }'
}

budget_color() {
    local pct="$1"
    if   ((pct >= 50)); then printf '%s' "$C_GREEN"
    elif ((pct >= 20)); then printf '%s' "$C_YELLOW"
    else printf '%s' "$C_RED"; fi
}

state_color() {
    case "$1" in
        train)   printf '%s' "$C_GREEN" ;;
        val)     printf '%s' "$C_CYAN" ;;
        warmup)  printf '%s' "$C_YELLOW" ;;
        error)   printf '%s' "$C_RED" ;;
        offline) printf '%s' "$C_RED" ;;
        idle)    printf '%s' "$C_DIM" ;;
        *)       printf '%s' "$C_BLUE" ;;
    esac
}

util_color() {
    local util="$1"
    if [[ ! "$util" =~ ^[0-9]+([.][0-9]+)?$ ]]; then printf '%s' "$C_DIM"; return; fi
    awk -v u="$util" -v g="$C_GREEN" -v y="$C_YELLOW" -v d="$C_DIM" -v r="$C_RED" \
        'BEGIN{if(u>=90)printf g; else if(u>=50)printf y; else if(u>0)printf d; else printf r}'
}

temp_color() {
    local temp="$1"
    if [[ ! "$temp" =~ ^[0-9]+([.][0-9]+)?$ ]]; then printf '%s' "$C_DIM"; return; fi
    awk -v t="$temp" -v g="$C_GREEN" -v y="$C_YELLOW" -v r="$C_RED" \
        'BEGIN{if(t>=80)printf r; else if(t>=70)printf y; else printf g}'
}

git_color() {
    local mod="${1:-0}" push="${2:-0}"
    if [[ "$mod" == "nogit" ]]; then printf '%s' "$C_RED"; return; fi
    if [[ "$mod" =~ ^[0-9]+$ && "$push" =~ ^[0-9]+$ ]]; then
        if ((mod == 0 && push == 0)); then printf '%s' "$C_GREEN"
        else printf '%s' "$C_YELLOW"; fi
    else
        printf '%s' "$C_DIM"
    fi
}

fmt_git() {
    local mod="${1:-}" push="${2:-}" out=""
    if [[ "$mod" == "nogit" ]]; then
        local n="${push:-0}"
        [[ "$n" =~ ^[0-9]+$ ]] && ((n > 0)) && printf '%s' "NO GIT ${n}res" || printf '%s' "NO GIT"
        return
    fi
    if [[ ! "$mod" =~ ^[0-9]+$ ]]; then printf '%s' "?"; return; fi
    [[ ! "$push" =~ ^[0-9]+$ ]] && push=0
    if ((mod == 0 && push == 0)); then printf '%s' "clean"; return; fi
    ((mod  > 0)) && out="${mod}m"
    ((push > 0)) && { [[ -n "$out" ]] && out+=" "; out+="${push}^"; }
    printf '%s' "$out"
}

build_bar() {
    local pct fill empty out="" i
    pct=$(clamp_percent "$1"); fill=$((pct / 5)); empty=$((20 - fill))
    for ((i=0; i<fill;  i++)); do out+="█"; done
    for ((i=0; i<empty; i++)); do out+="░"; done
    printf '%s' "$out"
}

table_rule() {
    printf '%s\n' "+--------------+----------+-------------+------------+------------+----------+--------+--------+---------+----------+----------+"
}

summary_rule() { printf '%s\n' "$(repeat_char "=" "$1")"; }

# ---- SSH helper (base64-encoded to avoid quoting issues) ----
ssh_run() {
    local port="$1" pass="$2" script="$3" b64
    b64=$(printf '%s' "$script" | base64 -w0)
    sshpass -p "$pass" ssh $SSH_OPTS -p "$port" root@"$HOST" "echo $b64 | base64 -d | bash" 2>/dev/null
}

# Combined fetch: step progress + GPU stats + git status in one SSH call
FETCH_SCRIPT='
# Check both /root/logs/ (scheduler location) and parameter-golf/logs/
f=$(ls -t /root/logs/*.txt /root/parameter-golf/logs/*.txt 2>/dev/null | head -1)
step=""
[ -n "$f" ] && step=$(tail -1 "$f")
gpu=$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null)
# Detect git repo; use "nogit" sentinel if not initialized
REPO=""
for d in /root/parameter-golf /root; do
    if git -C "$d" rev-parse --git-dir >/dev/null 2>&1; then REPO="$d"; break; fi
done
if [ -n "$REPO" ]; then
    git_mod=$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l | tr -d " ")
    git_push=$(git -C "$REPO" log origin/main..HEAD --oneline 2>/dev/null | wc -l | tr -d " ")
else
    # Count unsaved results as a proxy for "work not backed up"
    git_mod="nogit"
    git_push=$(ls -d /root/results/*/ /root/parameter-golf/results/*/ 2>/dev/null | wc -l | tr -d " ")
fi
printf "%s|||%s|||%s|||%s" "$step" "$gpu" "$git_mod" "$git_push"
'

# ---- Parallel fetch into temp files ----
declare -a TMP_FILES
for i in "${!G_NAMES[@]}"; do TMP_FILES+=("$(mktemp)"); done

cleanup() { rm -f "${TMP_FILES[@]}"; }
trap cleanup EXIT

for i in "${!G_NAMES[@]}"; do
    port="${G_PORTS[$i]}" pass="${G_PASSES[$i]}" tmp="${TMP_FILES[$i]}"
    ( ssh_run "$port" "$pass" "$FETCH_SCRIPT" > "$tmp" ) &
done
wait

# ---- Field extraction (format: step|||gpu|||git_mod|||git_push) ----
get_field() { awk -F'[|][|][|]' -v n="$2" 'NR==1{print $n}' "$1"; }

# ---- Progress parser ----
parse_progress() {
    local step_line="$1" cost_hr="$2" avg_ms
    P_STATE="idle"; P_STEP="-"; P_LOSS="-"; P_AVG="-"; P_COST1K="-"
    [[ -z "$step_line" ]] && return
    if   [[ "$step_line" =~ ^warmup_step:([0-9]+/[0-9]+) ]]; then
        P_STATE="warmup"; P_STEP="${BASH_REMATCH[1]}"
    elif [[ "$step_line" =~ ^step:([0-9]+/[0-9]+) ]]; then
        P_STEP="${BASH_REMATCH[1]}"
        if   [[ "$step_line" =~ val_loss:([0-9.]+)   ]]; then P_STATE="val";   P_LOSS="${BASH_REMATCH[1]}"
        elif [[ "$step_line" =~ train_loss:([0-9.]+) ]]; then P_STATE="train"; P_LOSS="${BASH_REMATCH[1]}"
        else P_STATE="step"; fi
    fi
    if [[ "$step_line" =~ step_avg:([0-9.]+)ms ]]; then
        avg_ms="${BASH_REMATCH[1]}"
        P_AVG=$(fmt_ms "$avg_ms")
        P_COST1K=$(awk -v cost="$cost_hr" -v ms="$avg_ms" 'BEGIN{printf "$%.3f", cost*ms/3600.0}')
    fi
}

# ---- GPU stats parser ----
parse_gpu_stats() {
    local gpu_line="$1" first_line util temp power
    G_ONLINE=0; G_ERROR=0; G_UTIL_RAW=""; G_TEMP_RAW=""
    G_UTIL="-"; G_TEMP="-"; G_POWER="-"
    [[ -z "$gpu_line" ]] && return
    first_line="${gpu_line%%$'\n'*}"
    IFS=',' read -r util temp power _ <<< "$first_line"
    util=$(trim "$util"); temp=$(trim "$temp"); power=$(trim "$power")
    [[ -z "$util" ]] && return
    if [[ ! "$util" =~ ^[0-9]+([.][0-9]+)?$ ]]; then G_ERROR=1; return; fi
    G_ONLINE=1; G_UTIL_RAW="$util"
    [[ "$temp" =~ ^[0-9]+([.][0-9]+)?$ ]] && G_TEMP_RAW="$temp"
    G_UTIL="${util}%"
    [[ -n "$G_TEMP_RAW" ]] && G_TEMP="${temp}C"
    G_POWER=$(fmt_watts "$power")
}

# ---- Row renderers ----
render_row() {
    local label="$1" cost_hr="$2" step_line="$3" gpu_line="$4" git_mod="$5" git_push="$6"
    local state state_label label_color state_fg util_fg temp_fg git_fg git_str

    parse_progress "$step_line" "$cost_hr"
    parse_gpu_stats "$gpu_line"

    if   ((G_ERROR   == 1)); then state="error";   G_UTIL="ERR"; G_TEMP="-"; G_POWER="-"
    elif ((G_ONLINE  == 0)); then state="offline";
        P_STEP="-"; P_LOSS="-"; P_AVG="-"; P_COST1K="-"; G_UTIL="-"; G_TEMP="-"; G_POWER="-"
    else state="$P_STATE"; fi

    case "$state" in
        train)   state_label="TRAIN"   ;;
        val)     state_label="VAL"     ;;
        warmup)  state_label="WARMUP"  ;;
        error)   state_label="ERROR"   ;;
        offline) state_label="OFFLINE" ;;
        idle)    state_label="IDLE"    ;;
        *)       state_label="STEP"    ;;
    esac

    label_color="$C_BOLD"
    state_fg=$(state_color "$state")
    if [[ "$state" == "error" ]]; then
        util_fg="$C_RED"; temp_fg="$C_RED"
    else
        util_fg=$(util_color "$G_UTIL_RAW"); temp_fg=$(temp_color "$G_TEMP_RAW")
    fi
    git_str=$(fmt_git "$git_mod" "$git_push")
    git_fg=$(git_color "$git_mod" "$git_push")

    printf "| %b%-12.12s%b | %b%-8.8s%b | %-11.11s | %-10.10s | %-10.10s | %-8.8s | %b%-6.6s%b | %b%-6.6s%b | %-7.7s | %b%-8.8s%b | %-8.8s |\n" \
        "$label_color" "$label" "$C_RESET" \
        "$state_fg" "$state_label" "$C_RESET" \
        "$P_STEP" "$P_LOSS" "$P_AVG" "$P_COST1K" \
        "$util_fg" "$G_UTIL" "$C_RESET" \
        "$temp_fg" "$G_TEMP" "$C_RESET" \
        "$G_POWER" \
        "$git_fg" "$git_str" "$C_RESET" \
        "\$$cost_hr"
}

render_row_compact() {
    local label="$1" cost_hr="$2" step_line="$3" gpu_line="$4" git_mod="$5" git_push="$6"
    local state state_label state_fg util_fg temp_fg git_fg git_str

    parse_progress "$step_line" "$cost_hr"
    parse_gpu_stats "$gpu_line"

    if   ((G_ERROR  == 1)); then state="error";   G_UTIL="ERR"; G_TEMP="-"; G_POWER="-"
    elif ((G_ONLINE == 0)); then state="offline";
        P_STEP="-"; P_LOSS="-"; P_AVG="-"; P_COST1K="-"; G_UTIL="-"; G_TEMP="-"; G_POWER="-"
    else state="$P_STATE"; fi

    case "$state" in
        train)   state_label="TRAIN"   ;;
        val)     state_label="VAL"     ;;
        warmup)  state_label="WARMUP"  ;;
        error)   state_label="ERROR"   ;;
        offline) state_label="OFFLINE" ;;
        idle)    state_label="IDLE"    ;;
        *)       state_label="STEP"    ;;
    esac

    state_fg=$(state_color "$state")
    util_fg=$(util_color "$G_UTIL_RAW"); temp_fg=$(temp_color "$G_TEMP_RAW")
    git_str=$(fmt_git "$git_mod" "$git_push")
    git_fg=$(git_color "$git_mod" "$git_push")

    printf "%b%-12.12s%b  %b%-7.7s%b  step %-11.11s  loss %-8.8s  avg %-8.8s  git %b%s%b\n" \
        "$C_BOLD" "$label" "$C_RESET" \
        "$state_fg" "$state_label" "$C_RESET" \
        "$P_STEP" "$P_LOSS" "$P_AVG" \
        "$git_fg" "$git_str" "$C_RESET"
    printf "  util %b%-6.6s%b  temp %b%-6.6s%b  power %-7.7s  \$/1k %-8.8s  \$%s/hr\n" \
        "$util_fg" "$G_UTIL" "$C_RESET" \
        "$temp_fg" "$G_TEMP" "$C_RESET" \
        "$G_POWER" "$P_COST1K" "$cost_hr"
}

# ---- Header ----
BAR=$(build_bar "$PCT")
BUDGET_FG=$(budget_color "$PCT")
HOURS_LEFT_DISPLAY=$(fmt_hours "$HOURS_LEFT")
SUMMARY_WIDTH=$FULL_TABLE_WIDTH
((TERM_COLS > 0 && TERM_COLS < SUMMARY_WIDTH)) && SUMMARY_WIDTH=$TERM_COLS

if ((USE_FULL_LAYOUT == 1)); then
    summary_rule "$SUMMARY_WIDTH"
    printf "%bGPU Fleet%b  %s UTC  |  uptime %s  |  %d GPUs  |  fleet \$%.2f/hr\n" \
        "$C_BOLD$C_CYAN" "$C_RESET" "$(date '+%Y-%m-%d %H:%M:%S')" "$ELAPSED_DISPLAY" "${#G_NAMES[@]}" "$TOTAL_HOURLY"
    printf "Budget left %b\$%.2f%b / \$%.2f  |  spent \$%.2f  |  burn \$%.4f/min  |  est left %sh\n" \
        "$BUDGET_FG" "$REMAINING" "$C_RESET" "$BUDGET" "$TOTAL_SPENT" "$COST_PER_MIN" "$HOURS_LEFT_DISPLAY"
    printf "Remaining   %b[%s]%b  %s%%\n" "$BUDGET_FG" "$BAR" "$C_RESET" "$PCT"
    summary_rule "$SUMMARY_WIDTH"

    table_rule
    printf "| %-12s | %-8s | %-11s | %-10s | %-10s | %-8s | %-6s | %-6s | %-7s | %-8s | %-8s |\n" \
        "GPU" "State" "Step" "Loss" "Avg step" "\$/1k" "Util" "Temp" "Power" "Git" "\$/hr"
    table_rule

    for i in "${!G_NAMES[@]}"; do
        render_row \
            "${G_LABELS[$i]}" "${G_COSTS[$i]}" \
            "$(get_field "${TMP_FILES[$i]}" 1)" \
            "$(get_field "${TMP_FILES[$i]}" 2)" \
            "$(get_field "${TMP_FILES[$i]}" 3)" \
            "$(get_field "${TMP_FILES[$i]}" 4)"
    done
    table_rule

else
    summary_rule "$SUMMARY_WIDTH"
    printf "%bGPU Fleet%b  %s UTC  (%d GPUs)\n" \
        "$C_BOLD$C_CYAN" "$C_RESET" "$(date '+%Y-%m-%d %H:%M:%S')" "${#G_NAMES[@]}"
    printf "uptime %s  |  fleet \$%.2f/hr  |  burn \$%.4f/min\n" "$ELAPSED_DISPLAY" "$TOTAL_HOURLY" "$COST_PER_MIN"
    printf "budget %b\$%.2f%b / \$%.2f  |  spent \$%.2f  |  left %sh (%s%%)\n" \
        "$BUDGET_FG" "$REMAINING" "$C_RESET" "$BUDGET" "$TOTAL_SPENT" "$HOURS_LEFT_DISPLAY" "$PCT"
    printf "%b[%s]%b\n" "$BUDGET_FG" "$BAR" "$C_RESET"
    summary_rule "$SUMMARY_WIDTH"

    for i in "${!G_NAMES[@]}"; do
        render_row_compact \
            "${G_LABELS[$i]}" "${G_COSTS[$i]}" \
            "$(get_field "${TMP_FILES[$i]}" 1)" \
            "$(get_field "${TMP_FILES[$i]}" 2)" \
            "$(get_field "${TMP_FILES[$i]}" 3)" \
            "$(get_field "${TMP_FILES[$i]}" 4)"
        printf '\n'
    done
fi
