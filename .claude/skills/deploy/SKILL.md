---
name: deploy
description: Deploy experiments to GPU fleet. Push a queue file or individual experiment to remote GPUs for execution. Use when the user wants to start experiments, send work to GPUs, or launch a queue of experiments.
---

# Deploy Experiments to GPU Fleet

Deploy experiment queues or individual experiments to remote GPU instances.

## Instructions

### If deploying a queue file (e.g., `/deploy queue_arch1_weightsharing.txt`):

1. Read the queue file from `lab/<filename>` in `/root/parameter-golf/`.
2. Show the user what experiments are in the queue (name, steps, env vars).
3. Check which experiments already have results in `results/<name>/train.log` (skip those).
4. Ask the user which GPU to target, or auto-select based on availability.
5. SSH into the target GPU and start the queue runner:

```bash
# From /root/parameter-golf on the remote GPU:
nohup bash lab/run_queue.sh lab/<queue_file> > logs/queue_runner.log 2>&1 &
```

6. Use credentials from `lab/gpu_creds.sh`. SSH pattern:
```bash
sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no -p $PORT root@$HOST "cd /root/parameter-golf && <command>"
```

### If deploying a single experiment (e.g., `/deploy my_test 500 MLP_ACT=relu2`):

1. Parse: name, steps, and optional ENV_VAR=value pairs from the arguments.
2. SSH into the target GPU and run:
```bash
nohup env <ENV_VARS> bash lab/run_experiment.sh <name> <steps> > logs/<name>.txt 2>&1 &
```

### Before deploying:

- Ensure the remote has the latest code: `git pull origin lab`
- Check no conflicting experiment is already running: `ps aux | grep train_gpt`
- Verify GPU is not already busy via `nvidia-smi`

### After deploying:

- Confirm the process started by checking `ps aux | grep train_gpt` on remote
- Tell the user they can monitor with `/status` or `/fleet`

## GPU Reference
- LOCAL 3090: local machine, no SSH needed
- REMOTE 3090: `$GPU_REMOTE3090_PORT`, `$GPU_REMOTE3090_PASS`
- L40S: `$GPU_L40S_PORT`, `$GPU_L40S_PASS`
- RTX 5090: `$GPU_5090_PORT`, `$GPU_5090_PASS`
- ARCH1-4: `$GPU_ARCH{1-4}_PORT`, `$GPU_ARCH{1-4}_PASS`
- All remotes use `root@$HOST`

## Key Files
- `lab/gpu_creds.sh` — SSH credentials
- `lab/run_experiment.sh` — Single experiment runner
- `lab/run_queue.sh` — Sequential queue runner
- `lab/run_queue_sync.sh` — Queue runner with git sync after each experiment
