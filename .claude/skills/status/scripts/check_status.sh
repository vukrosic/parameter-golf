#!/usr/bin/env bash
# Check detailed training status on all or a specific GPU.
# Usage: bash check_status.sh [gpu_name]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
FLEET_SCRIPTS="$REPO_ROOT/.claude/skills/fleet/scripts"

TARGET="${1:-all}"

check_gpu() {
    local name="$1" port="$2" pass="$3"

    echo "## $name"

    info=$(bash "$FLEET_SCRIPTS/ssh_run.sh" "$port" "$pass" '
        # Find running experiment
        exp_line=$(ps aux | grep train_gpt.py | grep -v grep | head -1)
        if [ -z "$exp_line" ]; then
            echo "STATUS=idle"
            exit 0
        fi

        pid=$(echo "$exp_line" | awk "{print \$2}")
        elapsed=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d " ")

        # Find the log file
        logfile=$(ls -t logs/*.txt 2>/dev/null | head -1)
        if [ -z "$logfile" ]; then
            echo "STATUS=running|PID=$pid|ELAPSED=$elapsed|NO_LOG=true"
            exit 0
        fi

        exp_name=$(basename "$logfile" .txt)

        # Parse latest metrics from log
        last_step=$(grep -oP "step \K[0-9]+" "$logfile" 2>/dev/null | tail -1)
        total_steps=$(grep -oP "step [0-9]+/\K[0-9]+" "$logfile" 2>/dev/null | tail -1)
        train_loss=$(grep -oP "train_loss[= :]+\K[0-9.]+" "$logfile" 2>/dev/null | tail -1)
        val_bpb=$(grep -oP "val_bpb[= :]+\K[0-9.]+" "$logfile" 2>/dev/null | tail -1)
        step_avg=$(grep -oP "step_avg[= :]+\K[0-9.]+" "$logfile" 2>/dev/null | tail -1)

        echo "STATUS=running|EXP=$exp_name|PID=$pid|ELAPSED=$elapsed|STEP=$last_step|TOTAL=$total_steps|TRAIN_LOSS=$train_loss|VAL_BPB=$val_bpb|STEP_AVG=$step_avg"
    ' 2>/dev/null) || info="STATUS=offline"

    status=$(echo "$info" | grep -oP 'STATUS=\K[a-z]+')

    if [ "$status" = "offline" ]; then
        echo "  Status: OFFLINE"
    elif [ "$status" = "idle" ]; then
        echo "  Status: IDLE (no experiment running)"
    else
        exp=$(echo "$info" | grep -oP 'EXP=\K[^|]+' || echo "unknown")
        elapsed=$(echo "$info" | grep -oP 'ELAPSED=\K[^|]+' || echo "?")
        step=$(echo "$info" | grep -oP 'STEP=\K[^|]+' || echo "?")
        total=$(echo "$info" | grep -oP 'TOTAL=\K[^|]+' || echo "?")
        tloss=$(echo "$info" | grep -oP 'TRAIN_LOSS=\K[^|]+' || echo "?")
        vbpb=$(echo "$info" | grep -oP 'VAL_BPB=\K[^|]+' || echo "?")
        savg=$(echo "$info" | grep -oP 'STEP_AVG=\K[^|]+' || echo "?")

        pct="?"
        eta="?"
        if [ "$step" != "?" ] && [ "$total" != "?" ] && [ -n "$total" ] && [ "$total" != "0" ]; then
            pct=$(python3 -c "print(f'{100*$step/$total:.1f}')" 2>/dev/null || echo "?")
            if [ "$savg" != "?" ] && [ -n "$savg" ]; then
                eta=$(python3 -c "remaining=$total-$step; ms=$savg; print(f'{remaining*ms/1000/60:.1f} min')" 2>/dev/null || echo "?")
            fi
        fi

        echo "  Experiment: $exp"
        echo "  Progress:   step $step/$total ($pct%)"
        echo "  Train loss: $tloss"
        echo "  Val BPB:    $vbpb"
        echo "  Step avg:   ${savg}ms"
        echo "  Wall time:  $elapsed"
        echo "  ETA:        ~$eta remaining"
    fi
    echo ""
}

echo "=== Training Status @ $(date '+%Y-%m-%d %H:%M:%S') ==="
echo ""

"$FLEET_SCRIPTS/discover_gpus.sh" | while read -r name port pass; do
    if [ "$TARGET" = "all" ] || echo "$name" | grep -qi "$TARGET"; then
        check_gpu "$name" "$port" "$pass"
    fi
done
