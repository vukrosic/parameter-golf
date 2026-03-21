---
name: deploy
description: Deploy experiments to GPU fleet. Push a queue file or individual experiment to remote GPUs for execution.
---

# Deploy Experiments to GPU Fleet

## Instructions

### Deploying a queue file (e.g., `/deploy queue_arch1_weightsharing.txt to ARCH1`)

1. Show the user what's in the queue file (`queues/<filename>`).
2. Check which experiments already have results in `results/<name>/train.log`.
3. Run the deploy script:
   ```bash
   bash .claude/skills/deploy/scripts/deploy_queue.sh <GPU_NAME> <queue_file>
   ```

### Deploying a single experiment (e.g., `/deploy my_test 500 MLP_ACT=relu2 to L40S`)

1. Parse the experiment name, steps, env vars, and target GPU from user args.
2. Run:
   ```bash
   bash .claude/skills/deploy/scripts/deploy_single.sh <GPU_NAME> <name> <steps> [ENV=val ...]
   ```

### If user doesn't specify a GPU

List available GPUs and their current status:
```bash
bash .claude/skills/fleet/scripts/fleet_status.sh
```
Then suggest the best idle GPU.

### After deploying
Tell the user they can monitor with `/status` or `/fleet`.

## Key Scripts
- `scripts/deploy_queue.sh` — Deploy a queue file to a GPU
- `scripts/deploy_single.sh` — Deploy a single experiment
- `.claude/skills/fleet/scripts/discover_gpus.sh` — Find available GPUs
