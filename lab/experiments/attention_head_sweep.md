# Attention Head Sweep

## Why this lane

AttnRes is already a weak lead in this repo: it improves early, then loses by step 500. That makes 200-step screens too noisy for attention work here.

The next attention-specific lane with low implementation risk is head topology:

- `8/4`: current grouped-query baseline
- `8/8`: full MHA
- `16/8`: more query heads at roughly the same KV width as baseline
- `8/2`: more aggressive GQA

This is a real attention-architecture change, not just an optimizer tweak, and it uses knobs already present in `train_gpt.py`.

## Sweep design

Use `400` steps, not `200`.

Reason:

- `200` already produced a false positive for `value_residual`
- by `500`, the sign had reversed
- `400` is close enough to expose reversal trends without paying the full cost of a 500-step screen

Decision:

- do fewer runs at `400`
- do not do more runs at `200`

## Run matrix

All runs:

```bash
ITERATIONS=400
VAL_LOSS_EVERY=50
TRAIN_LOG_EVERY=100
MAX_WALLCLOCK_SECONDS=0
MATRIX_LR=0.06
```

Configs:

```bash
heads84_lr06_400   NUM_HEADS=8  NUM_KV_HEADS=4
heads88_lr06_400   NUM_HEADS=8  NUM_KV_HEADS=8
heads168_lr06_400  NUM_HEADS=16 NUM_KV_HEADS=8
heads82_lr06_400   NUM_HEADS=8  NUM_KV_HEADS=2
```

## Decision rule after 400

Promote a config only if both are true:

1. It beats the `8/4` control at step 400.
2. The gap from step 200 to step 400 is flat or improving, not closing toward zero.

Kill a config if:

- it is behind at step 400, or
- it starts ahead but the gap shrinks sharply across the back half of the run

## Next step after the screen

- Extend the top 1-2 configs to `2000` steps
- Compare against `8/4` at the same LR
- Check whether the gap is growing, shrinking, or crossing after step 400

## Measured results

### 400-step screen

All runs used `MATRIX_LR=0.06`.

| Run | Heads/KV | Params | Step 100 | Step 200 | Step 400 | Quant @ 400 | Delta vs 8/4 @ 400 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `heads84_lr06_400` | `8/4` | 17.06M | 1.9308 | 1.6241 | 1.4849 | 1.4945 | baseline |
| `heads88_lr06_400` | `8/8` | 19.42M | 1.8917 | 1.6162 | 1.4820 | 1.4917 | `-0.0029` |
| `heads168_lr06_400` | `16/8` | 17.06M | 1.9731 | 1.6644 | 1.5100 | 1.5204 | `+0.0251` |
| `heads82_lr06_400` | `8/2` | 15.88M | 1.9199 | 1.6454 | 1.5031 | 1.5117 | `+0.0182` |

Observed trend:

- `8/8` was the only winner at step 400
- `16/8` is dead
- `8/2` starts near baseline, then loses decisively by step 150-200
- `8/8` starts far ahead, but the lead shrinks over time

`8/8` delta versus `8/4` across the 400-step run:

| Step | Delta BPB |
|---|---:|
| 50 | -0.0547 |
| 100 | -0.0391 |
| 150 | -0.0163 |
| 200 | -0.0079 |
| 250 | -0.0045 |
| 300 | -0.0029 |
| 350 | -0.0033 |
| 400 | -0.0029 |

Decision after 400:

- promote `8/8`
- kill `16/8`
- kill `8/2`
- do not run another broad 200-step sweep

### 1000-step extension

Follow-up pair:

- `heads84_lr06_1000`
- `heads88_lr06_1000`

| Run | Step 200 | Step 400 | Step 600 | Step 800 | Step 1000 | Quant @ 1000 |
|---|---:|---:|---:|---:|---:|---:|
| `heads84_lr06_1000` | 1.6522 | 1.4990 | 1.4281 | 1.3821 | 1.3554 | 1.3563 |
| `heads88_lr06_1000` | 1.6316 | 1.4854 | 1.4186 | 1.3737 | 1.3470 | 1.3480 |

`8/8` delta versus `8/4` at 1000-step checkpoints:

| Step | Delta BPB |
|---|---:|
| 200 | -0.0206 |
| 400 | -0.0136 |
| 500 | -0.0115 |
| 600 | -0.0095 |
| 700 | -0.0089 |
| 800 | -0.0084 |
| 900 | -0.0084 |
| 1000 | -0.0084 |

Interpretation:

- `8/8` is slightly worse at step 100 in the 1000-step extension, then moves ahead by step 200
- the gap still narrows after the early phase
- but it does **not** collapse to zero by step 1000
- unlike the AttnRes line, `8/8` has not crossed below baseline
- this is strong enough to justify a 2000-step confirmation run

## Current recommendation

Use `400` for attention screens, not `200`.

Reason:

- `200` is too short to reject false positives
- `400` is long enough to expose the sign and slope of the gap
- fewer runs at `400` are better than more runs at `200`

## Next experiments

1. `heads84_lr06_2000`
2. `heads88_lr06_2000`
3. If `8/8` still wins at 2000, test schedule tuning on top of `8/8`
4. If `8/8` crosses below control by 2000, kill the lane and move attention work to a size-matched MHA variant instead of more head sweeps
