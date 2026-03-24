---
name: status
description: Check detailed training status on one or all GPUs — live step count, loss curve, ETA, and recent log output.
---

# Experiment Status

Show detailed training progress for running experiments.

## Instructions

### All GPUs (default)
```bash
bash .claude/skills/status/scripts/check_status.sh
```

### Specific GPU (e.g., `/status ARCH1`)
```bash
bash .claude/skills/status/scripts/check_status.sh ARCH1
```

### If user asks for log output
Use the SSH helper to tail the log:
```bash
bash .claude/skills/fleet/scripts/ssh_run.sh <port> <pass> "tail -30 \$(ls -t logs/*.txt | head -1)"
```
Get port/pass from `bash .claude/skills/fleet/scripts/discover_gpus.sh`.

## Key Scripts
- `scripts/check_status.sh` — Detailed status for all or one GPU
- `.claude/skills/fleet/scripts/discover_gpus.sh` — GPU discovery
- `.claude/skills/fleet/scripts/ssh_run.sh` — SSH helper
