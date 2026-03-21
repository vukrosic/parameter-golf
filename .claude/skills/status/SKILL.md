---
name: status
description: Check detailed training status on one or all GPUs — live step count, loss curve, ETA, and recent log output. Use when the user wants to see how an experiment is progressing or check if training is still running.
---

# Experiment Status

Show detailed training progress for running experiments across the GPU fleet.

## Instructions

1. Source credentials from `lab/gpu_creds.sh` in `/root/parameter-golf/`.

2. For each GPU (or a specific one if the user specifies), SSH in and gather:

   a. **Running process**: `ps aux | grep train_gpt | grep -v grep` — extract PID, runtime, experiment name
   b. **Latest log lines**: Find the most recent log in `logs/` via `ls -t logs/*.txt | head -1`, then `tail -30` it
   c. **Training metrics**: Parse the last few lines for:
      - Current step / total steps
      - Latest train_loss and val_loss
      - val_bpb (bits per byte — the competition metric)
      - step_avg time in ms
      - Elapsed wall-clock time
   d. **ETA calculation**: From current step, total steps, and avg step time, compute time remaining

3. Format output as a per-GPU summary:
   ```
   ## GPU_NAME (HOSTNAME)
   Experiment: <name>
   Progress:   step 450/2000 (22.5%)
   Train loss: 3.456
   Val BPB:    1.2301 (last val at step 400)
   Step avg:   950ms
   ETA:        ~24 min remaining
   Wall time:  8 min elapsed
   ```

4. If no experiment is running on a GPU, report it as idle.

5. If the user asks for a specific experiment by name, search across all GPUs for it.

## SSH Pattern
```bash
sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -p $PORT root@$HOST "cd /root/parameter-golf && <cmd>"
```

## Key Files
- `lab/gpu_creds.sh` — SSH credentials
- Logs are in `logs/<experiment_name>.txt` on each GPU
- Results artifacts go to `results/<experiment_name>/`
