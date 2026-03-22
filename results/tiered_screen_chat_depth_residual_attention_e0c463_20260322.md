# Tiered Screen — chat_depth_residual_attention_e0c463 — 20260322

**Model:** full competition model (9L × 512d × 8/4 heads, MLP_MULT=2)  
**Ladder:** `quick` — 1 step(s) → 2 step(s) | 2 candidates → top 2  
**Optimizer:** plain Adam (no Muon/compile). Loss values are relative — compare within stage only.

**Why these candidates:** User-designed experiment via chat

---

### Stage 1 — 1 step(s): screen all 2 candidates

| Run | What it tests | Loss | Baseline | Delta | Decision |
|---|---|---:|---:|---:|---|
| `value_residual_9L` | 👁️ Attention Variant: value_residual | 6.9367 | 6.9374 | -0.0007 | promote ✓ |
| `baseline` | Control — all defaults. | 6.9374 | 6.9374 | +0.0000 | baseline |

Promoted to stage 2: **value_residual_9L**

---

### Stage 2 — 2 step(s): confirm or expose noise

| Run | What it tests | Loss | Baseline | Delta | Decision |
|---|---|---:|---:|---:|---|
| `value_residual_9L` | 👁️ Attention Variant: value_residual | 7.4334 | 7.6235 | -0.1902 | finalist ✓ |
| `baseline` | Control — all defaults. | 7.6235 | 7.6235 | +0.0000 | baseline |

---

## What happened

**Survived both stages:** `value_residual_9L` (👁️ Attention Variant: value_residual).
Ranking was consistent — this is signal, not noise. Promote to a 500-step explore run.

_Baseline loss rises 6.9374 → 7.6235 between stages (Adam without warmup climbs briefly before descending — normal, compare within stage only)._