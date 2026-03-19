# 8xH100 Submission Plan

## Best config found (1-GPU sweeps)

```
MATRIX_LR=30.0    # 750x default (was 0.04)
SCALAR_LR=3.0     # 75x default (was 0.04)
WARMUP_STEPS=0    # no warmup needed at this LR
```

Everything else default: 9x512, 8 heads, 4 KV heads, MLP_MULT=2, warmdown schedule, WARMDOWN_ITERS=1200.

## Run 1: Confirm best config (600s)

```bash
torchrun --nproc_per_node=8 train_gpt_golf.py
```
With env:
```
MATRIX_LR=30.0 SCALAR_LR=3.0 WARMUP_STEPS=0
MAX_WALLCLOCK_SECONDS=600 TRAIN_BATCH_TOKENS=524288
VAL_LOSS_EVERY=50 TRAIN_LOG_EVERY=10
RUN_ID=submit_v1
```

## Run 2-4: WARMDOWN_ITERS sweep (only thing we couldn't tune on 1 GPU)

Same as Run 1 but vary WARMDOWN_ITERS:
- Run 2: WARMDOWN_ITERS=400
- Run 3: WARMDOWN_ITERS=800
- Run 4: WARMDOWN_ITERS=2400

At 600s with 8xH100, we'll get ~180 steps. Warmdown kicks in at step (total - WARMDOWN_ITERS), so with ITERATIONS=100000 it effectively never triggers. Need to set ITERATIONS to actual expected steps (~180) for warmdown to work, OR adjust WARMDOWN_ITERS relative to ITERATIONS.

**IMPORTANT**: Check how WARMDOWN_ITERS interacts with ITERATIONS. If ITERATIONS=100000 but we stop at step 180 via wallclock, warmdown never activates. May need to set ITERATIONS=180 (estimated) to get proper schedule.

## Run 5: Refinement based on results

Pick best WARMDOWN_ITERS, optionally try:
- MATRIX_LR=20 or 40 (confirm 30 is optimal at full 600s)
- SCALAR_LR=2 or 5

## Expected outcome

Baseline: 1.2244 BPB. Our LR tuning gave ~3x improvement at 120s on 1-GPU. At 600s on 8xH100 with proper schedule, should beat baseline significantly.

## Risk: LR-schedule interaction

Our LR was tuned with early-training dynamics only (8-37 steps). At 180 steps the optimal LR might shift lower as loss landscape changes. If Run 1 shows instability in later steps, drop MATRIX_LR to 20.
