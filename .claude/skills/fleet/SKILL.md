---
name: fleet
description: Show GPU fleet status — all GPUs, their state (training/idle/offline), current experiment, step progress, loss, cost burn rate, and budget remaining. Use when the user wants to see what GPUs are doing, check fleet health, or monitor experiments.
---

# GPU Fleet Status

Show a live snapshot of all GPU instances in the parameter-golf fleet.

## Instructions

1. Source the GPU credentials from `lab/gpu_creds.sh` in the project root `/root/parameter-golf/`.

2. Run `bash lab/watch_all_gpus.sh` from `/root/parameter-golf/` to get the full fleet status table. This script already handles:
   - SSH into all remote GPUs in parallel
   - nvidia-smi stats (utilization, temp, power)
   - Training progress (step, loss, avg step time)
   - Cost tracking against $40 budget
   - Colorized output

3. Present the output to the user as-is (it's already well-formatted).

4. If the user asks about a specific GPU, also SSH into that machine to get more detail:
   - `tail -20` the latest log file in `logs/` to show recent training progress
   - Check what experiment is running via `ps aux | grep train_gpt`
   - Check disk usage with `df -h /root`

5. If any GPU shows as OFFLINE or ERROR, suggest troubleshooting steps:
   - Check if the instance is still running on Novita AI
   - Try SSH manually to see the error
   - Check if the experiment crashed (look at log tail)

## Key Files
- `lab/watch_all_gpus.sh` — Main monitoring script
- `lab/gpu_creds.sh` — SSH credentials (git-ignored)
- `lab/GPU_BENCHMARKS.md` — Performance reference
