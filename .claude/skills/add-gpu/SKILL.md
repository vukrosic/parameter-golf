---
name: add-gpu
description: Register a new GPU instance to the fleet. Walks through adding SSH credentials, testing connectivity, syncing the repo, and installing dependencies. Use when the user has a new GPU machine to add.
---

# Add GPU to Fleet

Interactively register a new GPU instance in the parameter-golf fleet.

## Instructions

### Step 1: Gather info from user
Ask for:
- **GPU name/label** (e.g., "A100", "ARCH5", "H100_spot")
- **SSH host** (usually same as existing: the Novita AI proxy host from `$HOST` in gpu_creds.sh)
- **SSH port**
- **SSH password**
- **Hourly cost** ($/hr)
- **GPU type** (for step timing estimates)

### Step 2: Test connectivity
```bash
sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $PORT root@$HOST "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
```
Verify the GPU is accessible and report its specs.

### Step 3: Add to gpu_creds.sh
Append new credential variables to `lab/gpu_creds.sh`:
```bash
# <GPU_LABEL> @ Novita AI
export GPU_<LABEL>_PORT=<port>
export GPU_<LABEL>_PASS="<password>"
```

### Step 4: Setup the remote
SSH into the new GPU and run:
```bash
# Clone/update the repo
cd /root
if [ -d parameter-golf ]; then
    cd parameter-golf && git pull origin lab
else
    git clone <repo_url> && cd parameter-golf && git checkout lab
fi

# Install dependencies
pip install -r requirements.txt

# Download data if needed
python3 data/cached_challenge_fineweb.py
```

### Step 5: Verify training works
Run a quick smoke test:
```bash
ITERATIONS=5 bash lab/run_experiment.sh smoke_test_<label> 5
```
Report the step timing and confirm training works.

### Step 6: Update monitoring
Tell the user they need to add the new GPU to `lab/watch_all_gpus.sh` for fleet monitoring.
Offer to do this — add a new cost variable, temp file, SSH fetch block, and render row.

### Step 7: Update sync
Similarly, add the new GPU to `lab/gpu_sync_cron.sh` for auto-sync.

## Key Files
- `lab/gpu_creds.sh` — Credential store (git-ignored)
- `lab/watch_all_gpus.sh` — Monitoring (needs manual update)
- `lab/gpu_sync_cron.sh` — Sync (needs manual update)
