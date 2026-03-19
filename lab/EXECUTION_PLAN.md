# Execution Plan

## Active lane: attention head topology

Current decision:

- use `400` steps for initial attention screens
- do fewer runs at `400`
- do not do more runs at `200`

Why:

- this repo already had a 200-step false positive in AttnRes
- 400-step screens were enough to separate durable leads from early-training noise

## Latest measured result

Best attention variant so far: full MHA, `NUM_HEADS=8 NUM_KV_HEADS=8`, `MATRIX_LR=0.06`

Key numbers:

| Run | Step 400 | Step 1000 | Quant @ 1000 |
|---|---:|---:|---:|
| `8/4` control | 1.4990 | 1.3554 | 1.3563 |
| `8/8` MHA | 1.4854 | 1.3470 | 1.3480 |
| Delta | -0.0136 | -0.0084 | -0.0083 |

Interpretation:

- the MHA lead shrinks as training continues
- but it remains positive through step 1000
- this is materially better than the AttnRes trend, which reversed sign by 500

## Immediate next run

Run only this pair next:

```bash
RUN_ID=heads84_lr06_2000 ITERATIONS=2000 VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=200 \
  CHECKPOINT_EVERY=200 MAX_WALLCLOCK_SECONDS=0 MATRIX_LR=0.06 \
  NUM_HEADS=8 NUM_KV_HEADS=4 python train_gpt.py

RUN_ID=heads88_lr06_2000 ITERATIONS=2000 VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=200 \
  CHECKPOINT_EVERY=200 MAX_WALLCLOCK_SECONDS=0 MATRIX_LR=0.06 \
  NUM_HEADS=8 NUM_KV_HEADS=8 python train_gpt.py
```

Decision rule after 2000:

- keep `8/8` only if it still beats `8/4`
- if the gap keeps shrinking but stays positive, try schedule tuning on `8/8`
- if the gap crosses below zero, kill this lane

## Killed configs

- `16/8`: loses badly by step 100
- `8/2`: near-flat early, then loses from step 150 onward
- AttnRes routing variants: earlier experiments already reversed by step 500
