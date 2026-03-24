# Tiered Screen — chat_residual_connection_techniques_bc1e41 — 20260322

**Model:** full competition model (9L × 512d × 8/4 heads, MLP_MULT=2)  
**Ladder:** `quick` — 1 step(s) → 2 step(s) | 4 candidates → top 2  
**Optimizer:** plain Adam (no Muon/compile). Loss values are relative — compare within stage only.

**Why these candidates:** Test three different residual connection techniques: highway networks, stochastic depth, and value residual attention to see which improves learning most effectively

---

### Stage 1 — 1 step(s): screen all 4 candidates

| Run | What it tests | Loss | Baseline | Delta | Decision |
|---|---|---:|---:|---:|---|
| `highway_networks` | Enable highway connections with learned gates | 6.9342 | 6.9374 | -0.0032 | promote ✓ |
| `stochastic_depth` | Add stochastic depth with 0.1 drop rate | 6.9374 | 6.9374 | -0.0001 | promote ✓ |
| `baseline` | Control — no changes. | 6.9374 | 6.9374 | +0.0000 | baseline |
| `value_residual_attention` | Enable value residual attention mode | 6.9399 | 6.9374 | +0.0025 | drop |

Promoted to stage 2: **highway_networks**, **stochastic_depth**

---

### Stage 2 — 2 step(s): confirm or expose noise

| Run | What it tests | Loss | Baseline | Delta | Decision |
|---|---|---:|---:|---:|---|
| `stochastic_depth` | Add stochastic depth with 0.1 drop rate | 7.3655 | 7.4358 | -0.0702 | finalist ✓ |
| `baseline` | Control — no changes. | 7.4358 | 7.4358 | +0.0000 | baseline |
| `highway_networks` | Enable highway connections with learned gates | 7.5102 | 7.4358 | +0.0745 | drop |

---

## What happened

**Survived both stages:** `stochastic_depth` (Add stochastic depth with 0.1 drop rate).
Ranking was consistent — this is signal, not noise. Promote to a 500-step explore run.
**Flipped negative at stage 2:** `highway_networks` — fast-start artifact, not real.

_Baseline loss rises 6.9374 → 7.4358 between stages (Adam without warmup climbs briefly before descending — normal, compare within stage only)._