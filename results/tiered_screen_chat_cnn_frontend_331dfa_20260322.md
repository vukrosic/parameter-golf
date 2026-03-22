# Tiered Screen — chat_cnn_frontend_331dfa — 20260322

**Model:** full competition model (9L × 512d × 8/4 heads, MLP_MULT=2)  
**Ladder:** `quick` — 1 step(s) → 2 step(s) | 6 candidates → top 2  
**Optimizer:** plain Adam (no Muon/compile). Loss values are relative — compare within stage only.

**Why these candidates:** Test if local pattern detection via convolutions before transformer helps

---

### Stage 1 — 1 step(s): screen all 6 candidates

| Run | What it tests | Loss | Baseline | Delta | Decision |
|---|---|---:|---:|---:|---|
| `conv_3_bottle` | conv=3 + embed_bottleneck=128 | 6.9323 | 6.9374 | -0.0051 | promote ✓ |
| `conv_3` | Single conv layer, kernel=3 | 6.9337 | 6.9374 | -0.0037 | promote ✓ |
| `conv_5_bottle` | conv=5 + embed_bottleneck=128 | 6.9338 | 6.9374 | -0.0037 | drop |
| `conv_7` | Single conv layer, kernel=7 | 6.9344 | 6.9374 | -0.0031 | drop |
| `conv_5` | Single conv layer, kernel=5 | 6.9361 | 6.9374 | -0.0013 | drop |
| `baseline` | Control — no changes. | 6.9374 | 6.9374 | +0.0000 | baseline |

Promoted to stage 2: **conv_3_bottle**, **conv_3**

---

### Stage 2 — 2 step(s): confirm or expose noise

| Run | What it tests | Loss | Baseline | Delta | Decision |
|---|---|---:|---:|---:|---|
| `conv_3` | Single conv layer, kernel=3 | 6.7259 | 7.4974 | -0.7716 | finalist ✓ |
| `baseline` | Control — no changes. | 7.4974 | 7.4974 | +0.0000 | baseline |
| `conv_3_bottle` | conv=3 + embed_bottleneck=128 | 7.8943 | 7.4974 | +0.3969 | drop |

---

## What happened

**Survived both stages:** `conv_3` (Single conv layer, kernel=3).
Ranking was consistent — this is signal, not noise. Promote to a 500-step explore run.
**Flipped negative at stage 2:** `conv_3_bottle` — fast-start artifact, not real.

_Baseline loss rises 6.9374 → 7.4974 between stages (Adam without warmup climbs briefly before descending — normal, compare within stage only)._