---
name: collect
description: Pull experiment results from all remote GPUs immediately. Syncs results directories, commits, and makes everything available locally. Use when the user wants to see results without waiting for the 2-hour cron sync.
---

# Collect Results from GPU Fleet

Immediately sync all experiment results from remote GPUs to the local machine.

## Instructions

1. Source credentials from `lab/gpu_creds.sh` in `/root/parameter-golf/`.

2. For each remote GPU, SSH in and:
   a. Stage result artifacts:
   ```bash
   git add results/*/submission.json results/*/config.json results/*/hparams.json results/*/README.md 2>/dev/null
   ```
   b. Commit if there are changes:
   ```bash
   git commit -m "results(<gpu_name>): auto-sync $(date +%Y-%m-%d_%H:%M)"
   ```
   c. Push to origin lab:
   ```bash
   git push origin lab
   ```

3. Locally, pull all changes:
   ```bash
   cd /root/parameter-golf && git pull origin lab
   ```

4. Stage and push any local results too.

5. Pull on all remotes so they're in sync (do this in parallel with `&` and `wait`).

6. Report to the user:
   - How many new experiment results were pulled
   - List the new experiments by name
   - Show the best val_bpb from the new results (parse from `results/*/submission.json`)

## Alternative: Use Existing Script
You can also just run the existing sync script:
```bash
bash lab/gpu_sync_cron.sh
```
This does all the above steps. Then summarize what was synced.

## Key Files
- `lab/gpu_sync_cron.sh` — Existing sync script
- `lab/gpu_creds.sh` — SSH credentials
- `results/*/submission.json` — Contains final metrics per experiment
