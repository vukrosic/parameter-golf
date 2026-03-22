# Tiered Screen — Layer Recursion/Reuse — 20260322

Ladder: ~10s (3 steps) → ~20s (6 steps) | Full competition model (9L × 512d)

---

### Stage 1 — ~10s

| Run | Architecture change | Loss | Baseline loss | Delta loss | Decision |
|---|---|---:|---:|---:|---|
| s1_recur_1b9c | 1 unique block × 9 cycles | 8.6191 | 8.9210 | -0.3019 | promote ✓ |
| s1_recur_3b3c | 3 unique blocks × 3 cycles | 8.6500 | 8.9210 | -0.2710 | promote ✓ |
| s1_recur_3b6c | 3 unique blocks × 6 cycles | 8.7046 | 8.9210 | -0.2164 | drop |
| s1_recur_9b2c | 9 unique blocks × 2 cycles | 8.9142 | 8.9210 | -0.0068 | drop |
| s1_recur_baseline | 9 layers, no reuse | 8.9210 | 8.9210 | +0.0000 | baseline |

---

### Stage 2 — ~20s

| Run | Architecture change | Loss | Baseline loss | Delta loss | Decision |
|---|---|---:|---:|---:|---|
| s2_recur_baseline | 9 layers, no reuse | 10.3928 | 10.3928 | +0.0000 | baseline |
| s2_recur_1b9c | 1 unique block × 9 cycles | 10.5350 | 10.3928 | +0.1423 | drop |
| s2_recur_3b3c | 3 unique blocks × 3 cycles | 11.4032 | 10.3928 | +1.0104 | drop |

---

## Conclusion

No candidate beat the baseline at stage 2. Layer recursion/reuse did not help at these step counts. Consider different recursion strategies or longer runs before concluding.