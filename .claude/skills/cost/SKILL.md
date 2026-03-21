---
name: cost
description: Show GPU fleet budget report — total spent, remaining budget, burn rate, per-GPU costs, and time until exhaustion.
---

# GPU Fleet Cost Report

## Instructions

Run the cost report:
```bash
bash .claude/skills/cost/scripts/cost_report.sh
```

This shows:
- Per-GPU hourly rate, status, session cost, and projected daily cost
- Total fleet burn rate
- Total spent this session
- Budget remaining (out of $40)
- Time until budget exhaustion
- Warning if budget is below 20%

### Cost-per-step reference
- RTX 3090: ~2700ms/step, $0.075/1k steps
- L40S: ~950ms/step, $0.074/1k steps
- RTX 5090: ~605ms/step, $0.050/1k steps

## Key Scripts
- `scripts/cost_report.sh` — Full cost report
- `infra/watch_all_gpus.sh` — Has cost tracking built in
