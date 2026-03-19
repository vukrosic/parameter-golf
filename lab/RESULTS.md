# Experiment Results (Measured)

All runs on 1xL40S, seed 1337, SP-1024 tokenizer, default config unless noted.
Lower BPB = better. Target: ≤1.2194 (beat 1.2244 by ≥0.005).

## Baseline Curves (500 steps)

```
Step | Baseline | VR Gated      Δ | VR Mid        Δ | Weighted      Δ | W.Vector      Δ
---------------------------------------------------------------------------------------
  50 |   2.4607 |   2.3834 -0.077 |   2.3790 -0.082 |   2.3677 -0.093 |   2.3606 -0.100
 100 |   1.9795 |   1.9222 -0.057 |   1.9420 -0.038 |   1.9503 -0.029 |   1.9488 -0.031
 150 |   1.7574 |   1.7050 -0.052 |   1.7220 -0.035 |   1.7300 -0.027 |   1.7249 -0.033
 200 |   1.6545 |   1.6212 -0.033 |   1.6313 -0.023 |   1.6367 -0.018 |   1.6317 -0.023
 250 |   1.5886 |   1.5691 -0.020 |   1.5746 -0.014 |   1.5780 -0.011 |   1.5758 -0.013
 300 |   1.5488 |   1.5311 -0.018 |   1.5370 -0.012 |   1.5380 -0.011 |   1.5375 -0.011
 350 |   1.5138 |   1.5013 -0.013 |   1.5072 -0.007 |   1.5079 -0.006 |   1.5090 -0.005
 400 |   1.4925 |   1.4816 -0.011 |   1.4864 -0.006 |   1.4872 -0.005 |   1.4876 -0.005
 450 |   1.4682 |   1.4626 -0.006 |   1.4677 -0.001 |   1.4677 -0.001 |   1.4690 +0.001
 500 |   1.4522 |   1.4529 +0.001 |   1.4578 +0.006 |   1.4579 +0.006 |   1.4594 +0.007
```

### Post-Quant (int8+zlib) at 500 steps

| Run | Pre-quant | Post-quant | Quant Gap |
|-----|-----------|------------|-----------|
| baseline_500 | 1.4522 | 1.4582 | 0.0060 |
| vr_gated_500 | 1.4529 | 1.4628 | 0.0099 |
| vr_mid_500 | 1.4578 | 1.4679 | 0.0101 |
| weighted_500 | 1.4579 | 1.4678 | 0.0099 |
| weighted_vec_500 | 1.4594 | 1.4689 | 0.0095 |

## AttnRes Conclusion

All 5 variants tested (cumsum, value_residual, value_residual_gated, weighted, weighted_vector).
All start better (faster initial learning) but baseline catches up and passes by step 500.
Trend is diverging (baseline pulling ahead). **AttnRes is dead.**

Weighted_vector (23K extra params) was the worst at step 500 (+0.007 pre-quant). More params = more noise at convergence.

## LR Sweep (200 steps, post-quant)

| MATRIX_LR | Post-quant BPB | Δ vs default |
|-----------|---------------|-------------|
| 0.06 | 1.6467 | **-0.0091** |
| 0.08 | 1.6511 | -0.0047 |
| 0.04 | 1.6558 | (default) |
| 0.12 | 1.6660 | +0.0102 |
| 0.02 | 1.7050 | +0.0492 |
| 0.16 | 1.6774 | +0.0216 |

**Best: MATRIX_LR=0.06.** Not yet confirmed at longer horizons.

## Early AttnRes Runs (200 steps)

| Run | Post-quant BPB |
|-----|---------------|
| attnres_baseline (none) | 1.6558 |
| attnres_cumsum | 1.6626 (+0.0068, worse) |
| attnres_value_residual | 1.6341 (-0.0217, but see 500-step reversal above) |

## Known Bugs

- Several runs hit the 600s wallclock cap because `MAX_WALLCLOCK_SECONDS=0` was not set.
  Affected: p2_vr_gated_500, p2_vr_mid_500, attnres_vr_500.
  **Always set `MAX_WALLCLOCK_SECONDS=0` for L40S experiments.**

## Reference Numbers

| Metric | Value |
|--------|-------|
| 8xH100 record (baseline) | 1.2244 BPB |
| 4h unlimited (post-quant) | 1.2074 BPB |
| 4h unlimited (pre-quant) | 1.1749 BPB |
| 4h quant gap | 0.0325 BPB |
| 500-step quant gap | 0.0060 BPB |
| Steps in 600s on 8xH100 | ~13,780 |
| Step time on 1xL40S | ~977ms |
| Default config | 9 layers, 512 dim, 8 heads, 4 KV heads |
