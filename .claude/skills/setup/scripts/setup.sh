#!/usr/bin/env bash
# Full setup for a fresh parameter-golf machine.
# Usage: bash .claude/skills/setup/scripts/setup.sh [--smoke-test]
set -euo pipefail
cd "$(dirname "$0")/../../../.."

echo "=== Parameter Golf Setup ==="
echo ""

# 1. Python dependencies
echo "[1/4] Installing Python dependencies..."
if command -v pip3 &>/dev/null; then
    pip3 install -r requirements.txt
elif command -v pip &>/dev/null; then
    pip install -r requirements.txt
else
    echo "ERROR: pip not found. Install Python first."
    exit 1
fi
echo "  Done."

# 2. Download data
echo "[2/4] Downloading FineWeb dataset (sp1024 variant)..."
if [ -d "data/datasets/fineweb10B_sp1024" ] && [ "$(ls data/datasets/fineweb10B_sp1024/*.bin 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  Data already exists, skipping. Pass --force-data to re-download."
else
    python3 data/cached_challenge_fineweb.py --variant sp1024
fi
echo "  Done."

# 3. Git hooks
echo "[3/4] Installing git hooks..."
if [ -f "lab/install_git_hooks.sh" ]; then
    bash lab/install_git_hooks.sh
else
    git config core.hooksPath .githooks 2>/dev/null || true
fi
echo "  Done."

# 4. Smoke test (optional)
if [[ "${1:-}" == "--smoke-test" ]]; then
    echo "[4/4] Running smoke test (5 steps)..."
    ITERATIONS=5 VAL_LOSS_EVERY=0 bash lab/run_experiment.sh setup_smoke_test 5
    echo "  Smoke test passed!"
    # Clean up smoke test artifacts
    rm -rf results/setup_smoke_test logs/setup_smoke_test.txt
else
    echo "[4/4] Smoke test skipped (pass --smoke-test to run)"
fi

echo ""
echo "=== Setup complete! ==="
echo "Next steps:"
echo "  - Run an experiment: lab/run_experiment.sh my_test 200"
echo "  - Set up GPU fleet: use /add-gpu skill"
echo "  - Check fleet status: use /fleet skill"
