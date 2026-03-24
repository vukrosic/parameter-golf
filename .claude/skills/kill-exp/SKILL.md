---
name: kill-exp
description: Stop a running experiment on a specific GPU and free it for new work.
---

# Kill Experiment

## Instructions

### Check what's running first
```bash
bash .claude/skills/kill-exp/scripts/kill_experiment.sh <GPU_NAME>
```
This shows the running process and recent log output without killing anything.

### Kill with confirmation
After showing the user what's running, ask for confirmation. Then:
```bash
bash .claude/skills/kill-exp/scripts/kill_experiment.sh <GPU_NAME> --force
```

### If user specifies an experiment name instead of GPU
Search across all GPUs:
```bash
bash .claude/skills/fleet/scripts/discover_gpus.sh | while read name port pass; do
    bash .claude/skills/fleet/scripts/ssh_run.sh "$port" "$pass" "ps aux | grep <exp_name> | grep -v grep"
done
```

### Important
- Always confirm with the user before killing
- If partial results exist, suggest `/collect` first
- Killed experiments can be re-run (run_queue.sh skips completed ones)

## Key Scripts
- `scripts/kill_experiment.sh` — Show running process and optionally kill
