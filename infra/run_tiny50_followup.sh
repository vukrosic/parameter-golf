#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

QUEUE="${QUEUE:-queues/tiny50_followup_best.txt}"
MODE="${1:-all}"
QUEUE_STEM="$(basename "${QUEUE}" .txt)"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_LOG="logs/queue_${QUEUE_STEM}_${STAMP}.log"
PROGRESS_JSON="logs/queue_progress_${QUEUE_STEM}.json"

SMOKE_EXPERIMENTS=(
  u01_vr_silu_ctrl
  u14_ebn64_vr_silu
  u28_ebn64_vr_relu
  u42_conv3_vr_silu_ebn64
)

run_verify() {
  python3 infra/run_queue_tiny.py "${QUEUE}" --verify
}

run_smoke() {
  for exp in "${SMOKE_EXPERIMENTS[@]}"; do
    python3 infra/run_queue_tiny.py "${QUEUE}" --only "${exp}"
  done
}

run_full() {
  echo "Queue log: ${RUN_LOG}"
  echo "Progress JSON: ${PROGRESS_JSON}"
  python3 infra/run_queue_tiny.py "${QUEUE}" --summary-every 5 | tee "${RUN_LOG}"
}

case "${MODE}" in
  dry-run)
    python3 infra/run_queue_tiny.py "${QUEUE}" --dry-run
    ;;
  verify)
    run_verify
    ;;
  smoke)
    run_smoke
    ;;
  run)
    run_full
    ;;
  all)
    run_verify
    run_smoke
    run_full
    ;;
  *)
    echo "Usage: infra/run_tiny50_followup.sh [dry-run|verify|smoke|run|all]" >&2
    exit 1
    ;;
esac
