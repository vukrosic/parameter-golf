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

## EMBED_LR sweep (full batch, 25s, MATRIX_LR=30)

No signal at 8 steps — all values 3.263–3.265. Needs more steps.

## Schedule sweep (full batch, 25s, MATRIX_LR=30, SCALAR_LR=3.0)

| Schedule | WARMDOWN_ITERS | Val BPB |
|---|---|---|
| warmdown | 500 | **2.992** |
| warmdown | 1200 (default) | 3.008 |
| warmdown | 100 | 3.234 |
| cosine | any | ~3.9 |

Cosine didn't help at full batch. Warmdown_500 marginal win but schedule is entangled with total run length — unreliable at 8 steps.

## Current best config

- MATRIX_LR=30.0
- SCALAR_LR=3.0
- WARMDOWN_ITERS=1200 (default, don't tune until longer runs)
- Everything else default

## Key findings

1. Default MATRIX_LR (0.04) is massively undertrained — 30.0 is optimal at full batch
2. SCALAR_LR similarly undertrained — 3.0 optimal (75x default)
3. LR rankings stable across 5s/10s/20s/40s at small batch
4. Schedule needs longer runs to tune properly
5. EMBED_LR insensitive at 8 steps
