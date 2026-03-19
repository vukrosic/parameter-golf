# Parameter Golf Sweep Results

Baseline: 1.2244 BPB (default config, 600s, 8xH100, 16MB model)

## MATRIX_LR sweep (small batch 65K, 40s, ~172 steps)

| MATRIX_LR | Val BPB |
|---|---|
| 0.01 | 2.747 |
| 0.02 | 2.605 |
| 0.04 (default) | 2.490 |
| 0.08 | 2.297 |
| 0.12 | 2.158 |
| 0.16 | 2.109 |
| 0.20 | 2.061 |
| 0.24 | 2.039 |
| 0.32 | 2.009 |
| 0.48 | 1.997 |
| **0.64** | **1.994** |
| 0.80 | 1.998 |
| 1.0 | 2.014 |
| 1.5 | 2.043 |
| 2.0 | 2.086 |

## MATRIX_LR sweep (full batch 524K, 25s, 8 steps)

| MATRIX_LR | Val BPB |
|---|---|
| 1.0 | 4.888 |
| 2.0 | 4.254 |
| 3.0 | 4.050 |
| 5.0 | 3.784 |
| 8.0 | 3.552 |
| 12.0 | 3.396 |
| 20.0 | 3.298 |
| **30.0** | **3.265** |
| 50.0 | 3.281 |
| 80.0 | 3.349 |
| 120.0 | 3.375 |

Note: optimal LR scaled ~50x from small→full batch (0.64 → 30.0), consistent with linear scaling.

## SCALAR_LR sweep (full batch, 25s, MATRIX_LR=30)

| SCALAR_LR | Val BPB |
|---|---|
| 0.04 (default) | 3.265 |
| 0.5 | 3.277 |
| 1.0 | 3.176 |
| 1.5 | 3.101 |
| 2.0 | 3.054 |
| 2.5 | 3.023 |
| **3.0** | **3.008** |
| 4.0 | 3.031 |
| 5.0 | 3.133 |

## EMBED_LR sweep (full batch, 120s, 37 steps, MATRIX_LR=30)

No signal — all values 2.6485–2.6489 across 0.1→30.0. Completely insensitive (tied embeddings).

## Warmup sweep (full batch, 120s, MATRIX_LR=30, SCALAR_LR=3.0)

| WARMUP_STEPS | Val BPB |
|---|---|
| **0** | **2.6447** |
| 1 | 2.6488 |
| 3 | 2.6487 |
| 5 | 2.6484 |
| 10 | 2.7688 |

WARMUP=0 marginally best. WARMUP=10 hurts (wastes steps at this LR).

## Architecture sweep (full batch, 50s, MATRIX_LR=30, SCALAR_LR=3.0)

| Config | Params | Steps | Val BPB |
|---|---|---|---|
| 6x640 | 17.9M | 8 | 3.031 |
| 7x576 | 16.9M | 13 | 2.842 |
| **9x512 (default)** | **17.0M** | **16** | **2.800** |
| 12x448 | 17.3M | 13 | 2.834 |
| 15x384 | 15.9M | 13 | 2.888 |

Default 9x512 wins — also fastest per step. Wider/shallower configs lose on step throughput.

## MLP_MULT sweep (full batch, 50s)

| MLP_MULT | Params | Val BPB |
|---|---|---|
| 1 | 12.3M | 3.126 |
| **2 (default)** | **17.0M** | **2.838** |
| 3 | 21.8M | 2.790 |
| 4 | 26.5M | 2.803 |

MLP=3 slightly better but exceeds 16MB param cap. Default MLP=2 best within budget.

## Misc toggles (full batch, 50s)

| Toggle | Val BPB | vs baseline 2.800 |
|---|---|---|
| SwiGLU | 2.887 | worse |
| ROPE_BASE=1000 | 2.797 | same |
| ROPE_BASE=100000 | 2.804 | same |
| LOGIT_SOFTCAP=50 | 2.804 | same |
| QK_GAIN_INIT=2.0 | 2.815 | same |

Nothing useful.

## Scaling check (full batch, 120s)

| Config | Val BPB |
|---|---|
| Default (MATRIX_LR=0.04, SCALAR_LR=0.04) | 8.359 |
| **Best (MATRIX_LR=30, SCALAR_LR=3, WARMUP=0)** | **2.645** |

LR tuning alone: 8.359 → 2.645 at 120s. Massive win.

## Best config for 8xH100 submission

```
MATRIX_LR=30.0
SCALAR_LR=3.0
WARMUP_STEPS=0
```
Everything else default (9x512, MLP_MULT=2, warmdown schedule, WARMDOWN_ITERS=1200).

## What didn't matter

- EMBED_LR (tied embeddings make it irrelevant)
- Architecture changes (default 9x512 is already optimal for throughput)
- SwiGLU, RoPE base, logit softcap, QK gain
- Warmup (0 vs 1 is noise)
