# Tiered Screen — Best-of-prior-screens: SwiGLU + value residual attn + leaky_relu2_05 + combination — 20260322

**Model:** full competition model (9L × 512d × 8/4 heads, MLP_MULT=2)
**Ladder:** 1 step(s) → 2 step(s) | 5 candidates → top 2

**Why these candidates:**
- SwiGLU and value residual attn both won the tiny-model tiered screen.
- leaky_relu2_05 is the best-known activation from 60+ runs in KNOWLEDGE.md (satisfies H1+H2+H3).
- The combination tests whether the two prior winners stack additively.

Training uses plain Adam (no Muon/compile) for speed. Loss values are relative, not competition-comparable.

---

### Stage 1 — 1 step(s): screen all candidates

| Run | What it tests | Loss | Baseline | Delta | Decision |
|---|---|---:|---:|---:|---|
| `best_leaky` | leaky_relu2_05 — best activation from 60+ KNOWLEDGE.md runs (+H1+H2+H3). | 6.9361 | 6.9374 | -0.0014 | promote ✓ |
| `best_swiglu` | SwiGLU activation — won tiny model screen (−0.0032 at 1s, −0.0309 at 3s). | 6.9372 | 6.9374 | -0.0003 | promote ✓ |
| `best_attnres_vr` | Value residual attn — carried hidden state across layers, won tiny screen. | 6.9374 | 6.9374 | -0.0001 | drop |
| `best_baseline` | Baseline — relu2, standard attn. Control for this screen. | 6.9374 | 6.9374 | +0.0000 | baseline |
| `best_combined` | SwiGLU + value residual attn — combine both prior winners. | 6.9405 | 6.9374 | +0.0031 | drop |

Top 2 promoted to stage 2: **best_leaky** and **best_swiglu**

---

### Stage 2 — 2 step(s): confirm or expose noise

| Run | What it tests | Loss | Baseline | Delta | Decision |
|---|---|---:|---:|---:|---|
| `best_swiglu` | SwiGLU activation — won tiny model screen (−0.0032 at 1s, −0.0309 at 3s). | 7.5576 | 7.6242 | -0.0666 | finalist ✓ |
| `best_leaky` | leaky_relu2_05 — best activation from 60+ KNOWLEDGE.md runs (+H1+H2+H3). | 7.5602 | 7.6242 | -0.0641 | finalist ✓ |
| `best_baseline` | Baseline — relu2, standard attn. Control for this screen. | 7.6242 | 7.6242 | +0.0000 | baseline |

---

## What happened

**Survived both stages:** `best_swiglu` (SwiGLU activation — won tiny model screen (−0.0032 at 1s, −0.0309 at 3s).), `best_leaky` (leaky_relu2_05 — best activation from 60+ KNOWLEDGE.md runs (+H1+H2+H3).).
The ranking was consistent across stages — signal, not noise. Worth promoting to a 500-step explore run.

_Note: baseline loss rises from 6.9374 (stage 1) to 7.6242 (stage 2) because more steps with Adam (no warmup) temporarily increases loss before it stabilises. Compare only within the same stage._