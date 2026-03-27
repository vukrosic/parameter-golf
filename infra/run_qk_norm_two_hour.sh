#!/usr/bin/env bash
# Run the planned 2-hour QK normalization sweep on a single GPU.
# Usage:
#   bash infra/run_qk_norm_two_hour.sh
# Optional:
#   RUN_STRETCH_L2=1 bash infra/run_qk_norm_two_hour.sh

set -euo pipefail
cd "$(dirname "$0")/.."

COMMON_ENV=(
    "SKIP_QUANT_EVAL=1"
    "VAL_LOSS_EVERY=500"
    "MAX_WALLCLOCK_SECONDS=600"
    "HARD_TIMEOUT_SECONDS=600"
    "TRAIN_LOG_EVERY=10"
)

run_one() {
    local name="$1"
    local steps="$2"
    shift 2
    local extra_env=("$@")

    echo
    echo ">>> Running ${name}"
    env "${COMMON_ENV[@]}" "${extra_env[@]}" infra/run_experiment.sh "$name" "$steps"
}

best_candidate() {
    python3 - <<'PY'
import json
from pathlib import Path

names = [
    "explore_qk_none_500",
    "explore_q_only_500",
    "explore_k_only_500",
    "explore_qk_gain_05_500",
    "explore_qk_gain_10_500",
    "explore_qk_gain_20_500",
]

best_name = None
best_val = None
for name in names:
    path = Path("results") / name / "summary.json"
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    last_eval = data.get("last_eval") or {}
    val = last_eval.get("val_bpb")
    if val is None:
        continue
    if best_val is None or val < best_val:
        best_name = name
        best_val = val

if best_name is None:
    raise SystemExit("No candidate summary.json files found.")
print(best_name)
PY
}

best_candidate_env() {
    case "$1" in
        explore_qk_none_500) printf '%s\n' "QK_NORM_MODE=none" ;;
        explore_q_only_500) printf '%s\n' "QK_NORM_MODE=q_only" ;;
        explore_k_only_500) printf '%s\n' "QK_NORM_MODE=k_only" ;;
        explore_qk_gain_05_500) printf '%s\n' "QK_GAIN_INIT=0.5" ;;
        explore_qk_gain_10_500) printf '%s\n' "QK_GAIN_INIT=1.0" ;;
        explore_qk_gain_20_500) printf '%s\n' "QK_GAIN_INIT=2.0" ;;
        *)
            echo "Unknown best-candidate mapping: $1" >&2
            return 1
            ;;
    esac
}

echo "============================================"
echo "QK Norm Two-Hour Sweep"
echo "Common env: ${COMMON_ENV[*]}"
echo "============================================"

run_one explore_qk_control_500 500
run_one explore_qk_none_500 500 "QK_NORM_MODE=none"
run_one explore_q_only_500 500 "QK_NORM_MODE=q_only"
run_one explore_k_only_500 500 "QK_NORM_MODE=k_only"
run_one explore_qk_gain_05_500 500 "QK_GAIN_INIT=0.5"
run_one explore_qk_gain_10_500 500 "QK_GAIN_INIT=1.0"
run_one explore_qk_gain_20_500 500 "QK_GAIN_INIT=2.0"

BEST_NAME="$(best_candidate)"
BEST_ENV="$(best_candidate_env "$BEST_NAME")"

echo
echo "Best first-pass candidate: ${BEST_NAME}"
echo "Retesting with SEED=42 using ${BEST_ENV}"

run_one explore_qk_best_retest_seed42 500 "SEED=42" "$BEST_ENV"

if [ "${RUN_STRETCH_L2:-0}" = "1" ]; then
    echo
    echo "RUN_STRETCH_L2=1 set, running optional L2 normalization stretch goal."
    run_one explore_qk_stretch_l2_500 500 "QK_NORM_MODE=l2"
fi

cat <<'EOF'

Sweep complete.

Suggested quick comparison:
  python3 infra/analyze.py

Relevant result folders:
  results/explore_qk_control_500/
  results/explore_qk_none_500/
  results/explore_q_only_500/
  results/explore_k_only_500/
  results/explore_qk_gain_05_500/
  results/explore_qk_gain_10_500/
  results/explore_qk_gain_20_500/
  results/explore_qk_best_retest_seed42/
EOF
