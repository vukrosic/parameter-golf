# Tiered Screen — activations — 20260322

**Model:** full competition model (9L × 512d × 8/4 heads, MLP_MULT=2)  
**Ladder:** `quick` — 1 step(s) → 2 step(s) | 3 candidates → top 2  
**Optimizer:** plain Adam (no Muon/compile). Loss values are relative — compare within stage only.

**Why these candidates:** Testing top activation candidates from KNOWLEDGE.md and prior tiny-model screens.

---

### Stage 1 — 1 step(s): screen all 3 candidates

| Run | What it tests | Loss | Baseline | Delta | Decision |
|---|---|---:|---:|---:|---|
| `swiglu` | SwiGLU — won tiny screen, honest gain at full. | 6.9372 | 6.9374 | -0.0003 | promote ✓ |
| `leaky` | leaky_relu2_05 — best in 60+ KNOWLEDGE.md runs. | 6.9374 | 6.9374 | -0.0001 | promote ✓ |
| `baseline` | Control — relu2, no changes. | 6.9374 | 6.9374 | +0.0000 | baseline |

Promoted to stage 2: **swiglu**, **leaky**

---

### Stage 2 — 2 step(s): confirm or expose noise

| Run | What it tests | Loss | Baseline | Delta | Decision |
|---|---|---:|---:|---:|---|
| `baseline` | Control — relu2, no changes. | 7.4380 | 7.4380 | +0.0000 | baseline |
| `swiglu` | SwiGLU — won tiny screen, honest gain at full. | 7.4603 | 7.4380 | +0.0223 | drop |
| `leaky` | leaky_relu2_05 — best in 60+ KNOWLEDGE.md runs. | 7.5602 | 7.4380 | +0.1221 | drop |

---

## What happened

**Nothing beat the baseline at stage 2.** Stage-1 gains were initialization-transient. Drop everything and try a different direction.
**Flipped negative at stage 2:** `swiglu`, `leaky` — fast-start artifact, not real.

_Baseline loss rises 6.9374 → 7.4380 between stages (Adam without warmup climbs briefly before descending — normal, compare within stage only)._