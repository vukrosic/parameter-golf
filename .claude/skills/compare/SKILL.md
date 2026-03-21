---
name: compare
description: Compare experiment results — show a ranked table of experiments by val_bpb, compare configurations, and identify what changes improved performance. Use when the user wants to analyze results, find the best experiment, or compare two experiments.
---

# Compare Experiment Results

Analyze and compare completed experiment results.

## Instructions

### Default (no args): Show ranked leaderboard

1. Scan all `results/*/submission.json` files in `/root/parameter-golf/results/`.
2. Parse each for: experiment name, final val_bpb, val_loss, steps completed, model size, wall time.
3. Sort by val_bpb (lower is better — this is the competition metric).
4. Display as a table:
   ```
   Rank  Experiment                    val_bpb   val_loss  Steps   Time
   1     act_leaky05_gradfloor_2000    1.2187    2.845     2000    112m
   2     baseline_2000                 1.2244    2.858     2000    114m
   ...
   ```

### With specific experiments (e.g., `/compare exp1 exp2`):

1. Read `results/<exp1>/submission.json` and `results/<exp2>/submission.json`.
2. Also read their `config.json` or `hparams.json` to find configuration differences.
3. Show side-by-side comparison:
   ```
   Config          exp1                exp2                Delta
   val_bpb         1.2244              1.2187              -0.0057 ✓
   MLP_ACT         gelu                leaky05_gradfloor   changed
   NUM_LAYERS      9                   9                   same
   ...
   ```
4. Highlight which config changes likely drove the improvement.

### With a pattern (e.g., `/compare act_*`):

1. Glob match `results/act_*/submission.json`.
2. Show all matching experiments ranked by val_bpb.
3. Identify trends (e.g., "activation functions with grad floor consistently outperform").

### Analysis mode (e.g., `/compare --analyze`):

1. Load all results.
2. Group by experiment family (prefix before last `_` + step count).
3. For each family, show best result and key config.
4. Identify the single most impactful hyperparameter change.

## Key Files
- `results/*/submission.json` — Final metrics (val_bpb, val_loss, model_size_bytes)
- `results/*/config.json` — Training configuration used
- `results/*/hparams.json` — Human-readable hyperparameters
- `lab/analyze.py` — Existing analysis tool (can parse and compare logs)
