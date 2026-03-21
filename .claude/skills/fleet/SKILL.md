---
name: fleet
description: Show GPU fleet status — all GPUs, their state (training/idle/offline), current experiment, step progress, loss, cost burn rate, and budget remaining.
---

# GPU Fleet Status

Show a live snapshot of all GPU instances in the parameter-golf fleet.

## Instructions

1. Run the fleet status script to get a quick overview:
   ```bash
   bash .claude/skills/fleet/scripts/fleet_status.sh
   ```

2. For richer monitoring with cost tracking and loss curves, run:
   ```bash
   bash infra/watch_all_gpus.sh
   ```
   Present the data to the user in a clean table.

3. If the user asks about a specific GPU, use the SSH helper to get detail:
   ```bash
   bash .claude/skills/fleet/scripts/ssh_run.sh <port> <pass> "tail -20 logs/*.txt; nvidia-smi"
   ```
   Get the port/pass by running `bash .claude/skills/fleet/scripts/discover_gpus.sh` and finding the matching GPU name.

4. If any GPU shows OFFLINE, suggest:
   - Check if the instance is still running on the cloud provider
   - Try SSH manually to see the error
   - Check if the experiment crashed (look at log tail)

## Key Scripts
- `scripts/fleet_status.sh` — Quick fleet overview
- `scripts/discover_gpus.sh` — List all GPUs from gpu_creds.sh dynamically
- `scripts/ssh_run.sh` — SSH into any GPU by port/pass
- `infra/watch_all_gpus.sh` — Full monitoring with cost tracking
