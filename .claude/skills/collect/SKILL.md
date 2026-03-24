---
name: collect
description: Pull experiment results from all remote GPUs immediately, without waiting for the 2-hour cron sync.
---

# Collect Results from GPU Fleet

## Instructions

Run the collect script:
```bash
bash .claude/skills/collect/scripts/collect_results.sh
```

This will:
1. SSH into each GPU and commit/push any new results
2. Pull everything locally
3. Push local results
4. Sync all remotes
5. Report how many new experiments were collected and their val_bpb scores

### Alternative: Use existing cron script
```bash
bash infra/gpu_sync_cron.sh
```
Then summarize what was synced.

## Key Scripts
- `scripts/collect_results.sh` — Full collection with summary
- `infra/gpu_sync_cron.sh` — Existing sync (same logic, less reporting)
