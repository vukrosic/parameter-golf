---
name: cost
description: Show GPU fleet budget report — total spent, remaining budget, burn rate, per-GPU costs, and time until budget exhaustion. Use when the user asks about costs, budget, spending, or how much money is left.
---

# GPU Fleet Cost Report

Generate a detailed cost analysis for the GPU fleet.

## Instructions

1. Read the budget state file at `/tmp/gpu_watch_start` to get session start time. If it doesn't exist, note that no session is being tracked.

2. Calculate costs using these hourly rates (from Novita AI):
   - LOCAL 3090: $0.10/hr
   - REMOTE 3090: $0.10/hr
   - L40S: $0.28/hr
   - RTX 5090: $0.30/hr
   - ARCH1-ARCH4: Check `lab/gpu_creds.sh` or `lab/watch_all_gpus.sh` for current rates

3. Compute and display:
   - **Session duration**: Time since tracking started
   - **Total fleet rate**: Sum of all active GPU hourly rates
   - **Total spent this session**: elapsed_hours × fleet_rate
   - **Budget remaining**: $40.00 - total_spent
   - **Burn rate**: $/minute and $/hour
   - **Time until budget exhaustion**: remaining / fleet_rate
   - **Per-GPU breakdown**: Individual cost per GPU, daily projected cost
   - **Cost per 1000 steps**: For each GPU type (using known step times)

4. Also calculate cost-efficiency metrics:
   - Cost per experiment (avg steps × cost_per_1k_steps)
   - Which GPU gives best BPB per dollar (if results are available)

5. If budget is below 20%, warn the user prominently.

## Cost Per 1K Steps Reference (with torch.compile)
- RTX 3090: ~2700ms/step → $0.075/1k steps
- L40S: ~950ms/step → $0.074/1k steps
- RTX 5090: ~605ms/step → $0.050/1k steps

## Key Files
- `lab/watch_all_gpus.sh` — Has cost constants and budget logic
- `/tmp/gpu_watch_start` — Session start timestamp
