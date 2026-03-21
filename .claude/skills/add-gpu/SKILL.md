---
name: add-gpu
description: Register a new GPU instance to the fleet. Tests connectivity, adds credentials, sets up the repo, and configures monitoring.
---

# Add GPU to Fleet

Interactively register a new GPU instance.

## Instructions

### Step 1: Gather info from user
Ask for:
- **GPU name/label** (e.g., "A100", "H100_SPOT") — will become `GPU_<LABEL>_PORT/PASS` in creds
- **SSH port**
- **SSH password**
- **Hourly cost** ($/hr) — for budget tracking
- **GPU type** — for step timing estimates

The SSH host is shared across all GPUs (read `$HOST` from `lab/gpu_creds.sh`).

### Step 2: Test connectivity
```bash
bash .claude/skills/add-gpu/scripts/add_gpu.sh <port> <password>
```

### Step 3: Add to gpu_creds.sh
Append to `lab/gpu_creds.sh`:
```bash
# <LABEL> (added YYYY-MM-DD)
GPU_<LABEL>_PORT=<port>
GPU_<LABEL>_PASS="<password>"
```

### Step 4: Setup the remote
Use the setup skill's remote script:
```bash
bash .claude/skills/setup/scripts/setup_remote.sh <port> <password>
```

### Step 5: Verify with smoke test
```bash
bash .claude/skills/fleet/scripts/ssh_run.sh <port> <password> \
    "ITERATIONS=5 VAL_LOSS_EVERY=0 bash lab/run_experiment.sh smoke_test 5"
```

### Step 6: Update cost tracking
Add the hourly rate to `.claude/skills/cost/scripts/cost_report.sh` RATES array.

### Step 7: Confirm
Run `/fleet` to verify the new GPU appears in the fleet status.

## Key Scripts
- `scripts/add_gpu.sh` — Test connectivity and report GPU specs
- `.claude/skills/setup/scripts/setup_remote.sh` — Full remote setup
- `.claude/skills/fleet/scripts/discover_gpus.sh` — Auto-discovers new GPU after creds added
