#!/usr/bin/env bash
# Setup a remote GPU instance via SSH.
# Usage: bash setup_remote.sh <port> <password> [repo_url]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source "$REPO_ROOT/infra/gpu_creds.sh"

PORT="$1"
PASS="$2"
REPO_URL="${3:-$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null || echo 'https://github.com/openai/parameter-golf.git')}"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o LogLevel=ERROR"

echo "Setting up remote GPU on port $PORT..."

sshpass -p "$PASS" ssh $SSH_OPTS -p "$PORT" root@"$HOST" bash -s "$REPO_URL" << 'REMOTE_SCRIPT'
REPO_URL="$1"
set -euo pipefail

echo "[1/4] Cloning/updating repo..."
cd /root
if [ -d parameter-golf ]; then
    cd parameter-golf
    git fetch origin lab 2>/dev/null
    git checkout lab 2>/dev/null
    git pull origin lab 2>/dev/null
else
    git clone "$REPO_URL" parameter-golf
    cd parameter-golf
    git checkout lab 2>/dev/null || true
fi

echo "[2/4] Installing dependencies..."
pip install -r requirements.txt 2>/dev/null || pip3 install -r requirements.txt

echo "[3/4] Downloading data..."
if [ -d "data/datasets/fineweb10B_sp1024" ]; then
    echo "  Data already exists, skipping."
else
    python3 data/cached_challenge_fineweb.py --variant sp1024
fi

echo "[4/4] Quick GPU check..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== Remote setup complete ==="
REMOTE_SCRIPT

echo "Done setting up port $PORT."
