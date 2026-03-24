# Tiered Screen — chat_safe_vs_moe_explorer_016cdb — 20260322

**Model:** full competition model (9L × 512d × 8/4 heads, MLP_MULT=2)  
**Ladder:** `quick` — 1 step(s) → 2 step(s) | 3 candidates → top 2  
**Optimizer:** plain Adam (no Muon/compile). Loss values are relative — compare within stage only.

**Why these candidates:** User-designed experiment via chat

---

### Stage 1 — 1 step(s): screen all 3 candidates

| Run | What it tests | Loss | Baseline | Delta | Decision |
|---|---|---:|---:|---:|---|
| `safe_bet` | ⚡ Activation Function: leaky05, ↔️ Model Width: dim384, 🧠 Mixture of Experts: moe4_d384, 📝 Embedding Strategy: untied_bn128 | 6.9315 | 6.9374 | -0.0060 | promote ✓ |
| `moe_explorer` | ⚡ Activation Function: leaky05, ↔️ Model Width: dim384, 🧠 Mixture of Experts: moe4_d384, 📝 Embedding Strategy: untied_bn128, 🛡️ Regularization: stoch_depth_02 | 6.9315 | 6.9374 | -0.0060 | promote ✓ |
| `baseline` | Control — all defaults. | 6.9374 | 6.9374 | +0.0000 | baseline |

Promoted to stage 2: **safe_bet**, **moe_explorer**

---

### Stage 2 — 2 step(s): confirm or expose noise

| Run | What it tests | Loss | Baseline | Delta | Decision |
|---|---|---:|---:|---:|---|
| `moe_explorer` | ⚡ Activation Function: leaky05, ↔️ Model Width: dim384, 🧠 Mixture of Experts: moe4_d384, 📝 Embedding Strategy: untied_bn128, 🛡️ Regularization: stoch_depth_02 | 7.2192 | 7.4380 | -0.2188 | finalist ✓ |
| `safe_bet` | ⚡ Activation Function: leaky05, ↔️ Model Width: dim384, 🧠 Mixture of Experts: moe4_d384, 📝 Embedding Strategy: untied_bn128 | 7.3062 | 7.4380 | -0.1318 | finalist ✓ |
| `baseline` | Control — all defaults. | 7.4380 | 7.4380 | +0.0000 | baseline |

---

## What happened

**Survived both stages:** `moe_explorer` (⚡ Activation Function: leaky05, ↔️ Model Width: dim384, 🧠 Mixture of Experts: moe4_d384, 📝 Embedding Strategy: untied_bn128, 🛡️ Regularization: stoch_depth_02), `safe_bet` (⚡ Activation Function: leaky05, ↔️ Model Width: dim384, 🧠 Mixture of Experts: moe4_d384, 📝 Embedding Strategy: untied_bn128).
Ranking was consistent — this is signal, not noise. Promote to a 500-step explore run.

_Baseline loss rises 6.9374 → 7.4380 between stages (Adam without warmup climbs briefly before descending — normal, compare within stage only)._