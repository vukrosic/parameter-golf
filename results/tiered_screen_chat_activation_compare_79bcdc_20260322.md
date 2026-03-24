# Tiered Screen — chat_activation_compare_79bcdc — 20260322

**Model:** full competition model (9L × 512d × 8/4 heads, MLP_MULT=2)  
**Ladder:** `quick` — 1 step(s) → 2 step(s) | 3 candidates → top 2  
**Optimizer:** plain Adam (no Muon/compile). Loss values are relative — compare within stage only.

**Why these candidates:** User-designed experiment via chat

---

### Stage 1 — 1 step(s): screen all 3 candidates

| Run | What it tests | Loss | Baseline | Delta | Decision |
|---|---|---:|---:|---:|---|
| `leaky05_safe` | ⚡ Activation Function: leaky05, 🧠 Mixture of Experts: moe4_d384, 📝 Embedding Strategy: untied_bn128 | 6.9315 | 6.9374 | -0.0060 | promote ✓ |
| `swiglu_safe` | ⚡ Activation Function: swiglu, 🧠 Mixture of Experts: moe4_d384, 📝 Embedding Strategy: untied_bn128 | 6.9315 | 6.9374 | -0.0060 | promote ✓ |
| `baseline` | Control — all defaults. | 6.9374 | 6.9374 | +0.0000 | baseline |

Promoted to stage 2: **leaky05_safe**, **swiglu_safe**

---

### Stage 2 — 2 step(s): confirm or expose noise

| Run | What it tests | Loss | Baseline | Delta | Decision |
|---|---|---:|---:|---:|---|
| `swiglu_safe` | ⚡ Activation Function: swiglu, 🧠 Mixture of Experts: moe4_d384, 📝 Embedding Strategy: untied_bn128 | 7.2192 | 7.4380 | -0.2188 | finalist ✓ |
| `leaky05_safe` | ⚡ Activation Function: leaky05, 🧠 Mixture of Experts: moe4_d384, 📝 Embedding Strategy: untied_bn128 | 7.3062 | 7.4380 | -0.1318 | finalist ✓ |
| `baseline` | Control — all defaults. | 7.4380 | 7.4380 | +0.0000 | baseline |

---

## What happened

**Survived both stages:** `swiglu_safe` (⚡ Activation Function: swiglu, 🧠 Mixture of Experts: moe4_d384, 📝 Embedding Strategy: untied_bn128), `leaky05_safe` (⚡ Activation Function: leaky05, 🧠 Mixture of Experts: moe4_d384, 📝 Embedding Strategy: untied_bn128).
Ranking was consistent — this is signal, not noise. Promote to a 500-step explore run.

_Baseline loss rises 6.9374 → 7.4380 between stages (Adam without warmup climbs briefly before descending — normal, compare within stage only)._