# Tiered Screen — chat_layer_recursion_weight_sharing_21d721 — 20260322

**Model:** full competition model (9L × 512d × 8/4 heads, MLP_MULT=2)  
**Ladder:** `quick` — 1 step(s) → 2 step(s) | 5 candidates → top 2  
**Optimizer:** plain Adam (no Muon/compile). Loss values are relative — compare within stage only.

**Why these candidates:** Test optimal balance between weight sharing (num_unique_blocks) and repetition cycles (num_cycles) to reduce parameters while maintaining performance

---

### Stage 1 — 1 step(s): screen all 5 candidates

| Run | What it tests | Loss | Baseline | Delta | Decision |
|---|---|---:|---:|---:|---|
| `share_3x3` | 3 unique blocks, 3 cycles (9L total) - moderate weight sharing | 6.9349 | 6.9374 | -0.0025 | promote ✓ |
| `baseline` | Control — no changes. | 6.9374 | 6.9374 | +0.0000 | baseline |
| `share_4x2` | 4 unique blocks, 2 cycles (8L total) - light weight sharing | 6.9381 | 6.9374 | +0.0007 | promote ✓ |
| `share_1x9` | 1 unique block, 9 cycles (9L total) - extreme weight sharing | 6.9394 | 6.9374 | +0.0020 | drop |
| `share_2x4` | 2 unique blocks, 4 cycles (8L total) - aggressive weight sharing | 6.9423 | 6.9374 | +0.0049 | drop |

Promoted to stage 2: **share_3x3**, **share_4x2**

---

### Stage 2 — 2 step(s): confirm or expose noise

| Run | What it tests | Loss | Baseline | Delta | Decision |
|---|---|---:|---:|---:|---|
| `baseline` | Control — no changes. | 7.6242 | 7.6242 | +0.0000 | baseline |
| `share_3x3` | 3 unique blocks, 3 cycles (9L total) - moderate weight sharing | 7.6860 | 7.6242 | +0.0618 | drop |
| `share_4x2` | 4 unique blocks, 2 cycles (8L total) - light weight sharing | 7.7411 | 7.6242 | +0.1169 | drop |

---

## What happened

**Nothing beat the baseline at stage 2.** Stage-1 gains were initialization-transient. Drop everything and try a different direction.
**Flipped negative at stage 2:** `share_3x3`, `share_4x2` — fast-start artifact, not real.

_Baseline loss rises 6.9374 → 7.6242 between stages (Adam without warmup climbs briefly before descending — normal, compare within stage only)._