# Lab Setup

## Development Hardware

- **1x NVIDIA L40S** (48GB VRAM)
- All experiments run single-GPU with grad_accum=8 (automatic)
- Step dynamics identical to 8xH100 (same effective batch)

## Running an Experiment

Every experiment is a single invocation of `train_gpt.py` (or a fork) with env vars.
Use `lab/run_experiment.sh` for standardized runs:

```bash
# Quick sanity check (50 steps, ~2.5 min on L40S)
lab/run_experiment.sh my_test_name 50

# Medium run (200 steps, ~11 min)
lab/run_experiment.sh my_test_name 200

# Full L40S equivalent of 600s 8xH100 (13780 steps ≈ 12.7 hours)
lab/run_experiment.sh my_test_name 13780

# With custom env overrides
MATRIX_LR=0.08 NUM_LAYERS=12 MODEL_DIM=448 lab/run_experiment.sh my_arch_test 200
```

## Experiment Duration Guide

On 1xL40S at ~3.33s/step:

| Steps | Wall-clock | Equivalent 8xH100 time | Use for |
|-------|-----------|------------------------|---------|
| 15 | ~50s | ~0.65s | Smoke test only |
| 50 | ~2.8 min | ~2.2s | Quick architecture sanity |
| 200 | ~11 min | ~8.7s | LR/schedule comparison |
| 500 | ~28 min | ~22s | Medium convergence check |
| 1000 | ~55 min | ~43s | Solid convergence signal |
| 3000 | ~2.8 hr | ~2.2 min | Strong signal |
| 5000 | ~4.6 hr | ~3.6 min | Near-final validation |
| 13780 | ~12.7 hr | 600s (full run) | Final pre-submission |

## What to Measure

Every experiment log contains:
- `train_loss` at each logged step
- `val_loss` and `val_bpb` at validation intervals
- `step_avg` (ms/step, useful for throughput calibration)

For submission candidates, the script also produces:
- `final_model.int8.ptz` — the compressed artifact
- `final_int8_zlib_roundtrip val_bpb` — the actual submission score

## Choosing Step Counts for Experiments

The key insight: **you don't need to run all 13,780 steps to compare configs**.

Loss curves are monotonic and largely parallel during mid-training. If config A beats
config B at step 500, it will almost certainly beat it at step 13,780 (barring schedule
interactions). Use this:

1. **Screening** (50-200 steps): Compare 5-10 configs. Keep top 3.
2. **Validation** (500-1000 steps): Compare top 3. Confirm ranking holds.
3. **Final** (3000-5000 steps): Run best 1-2 configs. Check for late divergence.
4. **Pre-submission** (13780 steps): Run the winner once. Verify BPB target.

## Step-Based Scheduling for 8xH100

The warmdown schedule is time-based by default (uses wall-clock remaining).
For 8xH100 at ~43ms/step over 600s:

- Total steps: ~13,780
- warmdown_iters=1200 → warmdown starts at step ~12,580
- warmdown_iters=2400 → warmdown starts at step ~11,380

The time-based schedule auto-adapts, so this transfers correctly as long as
MAX_WALLCLOCK_SECONDS=600 is set on the H100 run. No manual step calibration needed.

## Interpreting Results

**val_bpb is the only metric that matters for the challenge.**

- val_loss is in nats (natural log). val_bpb converts to bits-per-byte.
- train_loss tracks optimization but can diverge from val due to overfitting.
- The quant gap (pre-quant BPB minus post-quant BPB) is ~0.03 BPB for the baseline.
  Anything that reduces this gap is free performance.
