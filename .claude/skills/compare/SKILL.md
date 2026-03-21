---
name: compare
description: Compare experiment results — ranked table by val_bpb, side-by-side config diffs, or pattern-matched analysis.
---

# Compare Experiment Results

## Instructions

### Default (no args): Ranked leaderboard
```bash
bash .claude/skills/compare/scripts/compare_results.sh
```

### Two experiments (e.g., `/compare exp1 exp2`)
```bash
bash .claude/skills/compare/scripts/compare_results.sh exp1 exp2
```
Shows side-by-side config comparison with deltas.

### Pattern match (e.g., `/compare act_*`)
Read all `results/act_*/submission.json` files, rank by val_bpb, and identify trends.

### Deep analysis (e.g., `/compare --analyze`)
Load all results, group by experiment family (prefix), identify the most impactful hyperparameter change.

## Key Scripts
- `scripts/compare_results.sh` — Leaderboard and side-by-side comparison
- `infra/analyze.py` — Existing analysis tool for deeper log parsing

## Key Files
- `results/*/submission.json` — Final metrics (val_bpb, val_loss)
- `results/*/config.json` — Training configuration
- `results/*/hparams.json` — Human-readable hyperparameters
