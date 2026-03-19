# AttnRes Experiment Plan v2 — Scaled Experiments

**Status**: Phase 1 complete (A: cumsum, B: value_residual at 200 & 500 steps). Key finding: value residual is worse but gap is closing. Need longer runs and smarter variants.

## Key Finding from Phase 1

Value residual at 500 steps: consistently worse than baseline, but gap narrows from 0.037 BPB (step 100) to 0.004 BPB (step 500). Logarithmic convergence suggests **crossover around step 1500-2000**. This means value residuals may help at the H100's 13,780-step horizon.

Cumsum variant: worse at 200 steps (1.6626 vs 1.6558 baseline). Not extended.

## Phase 2: Confirm Crossover (Priority: CRITICAL)

### Experiment 2A: Value Residual Extended to 2000 Steps

**Goal**: Find the crossover point where value residual beats baseline.

```bash
# Baseline 2000 steps
RUN_ID=baseline_2000 ITERATIONS=2000 VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=100 \
  MAX_WALLCLOCK_SECONDS=0 python train_gpt.py

# Value residual 2000 steps
RUN_ID=vr_2000 ATTNRES_MODE=value_residual ITERATIONS=2000 VAL_LOSS_EVERY=100 \
  TRAIN_LOG_EVERY=100 MAX_WALLCLOCK_SECONDS=0 python train_gpt.py
```

- **Wall time on L40S**: ~33 min each (2000 steps * ~1s/step)
- **Disable wallclock cap**: `MAX_WALLCLOCK_SECONDS=0` so we hit exact step count
- **Success**: crossover observed (VR beats baseline at any checkpoint >= 1000)
- **Fail**: gap plateaus or widens. Kill the line, move to Phase 3.

### Experiment 2B: Value Residual at Best LR (0.06)

Phase 1 LR sweep showed MATRIX_LR=0.06 beats 0.04. Value residual experiments all used 0.04. The gap dynamics may differ at higher LR.

```bash
# Baseline with best LR
RUN_ID=baseline_2000_lr06 MATRIX_LR=0.06 ITERATIONS=2000 VAL_LOSS_EVERY=100 \
  TRAIN_LOG_EVERY=100 MAX_WALLCLOCK_SECONDS=0 python train_gpt.py

# Value residual with best LR
RUN_ID=vr_2000_lr06 ATTNRES_MODE=value_residual MATRIX_LR=0.06 ITERATIONS=2000 \
  VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=100 MAX_WALLCLOCK_SECONDS=0 python train_gpt.py
```

Run in parallel with 2A if GPU budget allows.

---

## Phase 3: Smarter Routing Variants (Priority: HIGH)

These address the core problem: layer 0's values are noisy early, so blindly adding them hurts. Each variant fixes this differently.

### Experiment 3A: Gated Value Residual

Add a learnable scalar gate per layer, initialized to 0 (no residual at init). The model learns when to use the value residual as training progresses.

```python
# In CausalSelfAttention.__init__():
self.v_gate = nn.Parameter(torch.zeros(1))  # init to 0 = bypass

# In forward():
if v_residual is not None:
    v = v + torch.sigmoid(self.v_gate) * v_residual
```

- **Extra params**: 9 scalars (negligible)
- **Hypothesis**: gate stays ~0 early (avoiding the noise penalty), opens as layer-0 values become useful
- **Run at**: 500 and 2000 steps, MATRIX_LR=0.04 and 0.06

```bash
RUN_ID=vr_gated_2000 ATTNRES_MODE=value_residual_gated ITERATIONS=2000 \
  VAL_LOSS_EVERY=100 MAX_WALLCLOCK_SECONDS=0 python train_gpt.py
```

### Experiment 3B: Value Residual from Middle Layer (layer N/2)

Instead of layer 0 (raw embeddings), use layer 4 (middle of encoder). By layer 4, representations are richer, so the residual should help earlier.

```python
# In GPT.forward():
# Capture V from layer num_encoder_layers//2 instead of layer 0
capture_layer = self.num_encoder_layers // 2  # = 2 for 9-layer model
```

- **Extra params**: 0
- **Hypothesis**: mid-layer values are more useful than layer-0 values
- **Run at**: 500 and 2000 steps

```bash
RUN_ID=vr_mid_2000 ATTNRES_MODE=value_residual_mid ITERATIONS=2000 \
  VAL_LOSS_EVERY=100 MAX_WALLCLOCK_SECONDS=0 python train_gpt.py
```

### Experiment 3C: Weighted Inter-Layer Attention (PLAN.md Variant C)

Each layer learns a softmax-weighted combination of ALL previous layer outputs. This is the most principled variant — it subsumes resid_mix and skip_weights.

```python
self.layer_weights = nn.ParameterList([
    nn.Parameter(torch.zeros(i + 1))
    for i in range(num_layers)
])
```

- **Extra params**: sum(1..9) = 45 scalars
- **VRAM**: Store all 9 layer outputs (~9 * batch * seq * 512 = ~2.4GB extra)
- **Run at**: 500 and 2000 steps
- **Note**: May need to remove resid_mix/skip_weights to avoid redundancy

```bash
RUN_ID=weighted_2000 ATTNRES_MODE=weighted ITERATIONS=2000 \
  VAL_LOSS_EVERY=100 MAX_WALLCLOCK_SECONDS=0 python train_gpt.py
```

### Experiment 3D: Per-Dimension Weighted Routing (PLAN.md Variant D)

Like 3C but with vector weights per layer pair. Different dimensions can route from different source layers.

```python
self.layer_weights = nn.ParameterList([
    nn.Parameter(torch.zeros(i + 1, model_dim))
    for i in range(num_layers)
])
```

- **Extra params**: 23,040 (~23KB, well within 140KB headroom)
- **Run at**: 500 and 2000 steps

```bash
RUN_ID=weighted_vec_2000 ATTNRES_MODE=weighted_vector ITERATIONS=2000 \
  VAL_LOSS_EVERY=100 MAX_WALLCLOCK_SECONDS=0 python train_gpt.py
```

---

## Phase 4: Scale Validation (Priority: MEDIUM)

Extend winners from Phase 2-3 to H100-relevant step counts.

### Experiment 4A: Winner at 5000 Steps

```bash
RUN_ID=winner_5000 ATTNRES_MODE=<winner> ITERATIONS=5000 VAL_LOSS_EVERY=250 \
  TRAIN_LOG_EVERY=500 MAX_WALLCLOCK_SECONDS=0 python train_gpt.py
```

- **Wall time on L40S**: ~83 min
- **Purpose**: Confirm improvement holds at 36% of H100 horizon

### Experiment 4B: Winner at 10000 Steps

```bash
RUN_ID=winner_10000 ATTNRES_MODE=<winner> ITERATIONS=10000 VAL_LOSS_EVERY=500 \
  TRAIN_LOG_EVERY=1000 MAX_WALLCLOCK_SECONDS=0 python train_gpt.py
```

- **Wall time on L40S**: ~2.8 hours
- **Purpose**: Confirm improvement holds at 73% of H100 horizon

### Experiment 4C: Winner + Deeper Architecture (12 layers)

If AttnRes helps routing, it should enable deeper models within the same parameter budget.

```bash
RUN_ID=winner_12L_5000 ATTNRES_MODE=<winner> NUM_LAYERS=12 MODEL_DIM=448 \
  ITERATIONS=5000 VAL_LOSS_EVERY=250 MAX_WALLCLOCK_SECONDS=0 python train_gpt.py
```

- 12 layers x 448 dim stays within 16MB cap
- **Hypothesis**: AttnRes + more layers > baseline 9-layer at same parameter count

---

## Phase 5: Full Horizon Validation

### Experiment 5A: Winner at 13,780 Steps (H100 Equivalent)

```bash
RUN_ID=winner_full ATTNRES_MODE=<winner> MATRIX_LR=<best> ITERATIONS=13780 \
  VAL_LOSS_EVERY=1000 TRAIN_LOG_EVERY=1000 MAX_WALLCLOCK_SECONDS=0 python train_gpt.py
```

- **Wall time on L40S**: ~3.8 hours
- **Target**: val_bpb <= 1.2194 (int8+zlib roundtrip)
- **Must verify**: artifact size <= 16MB

---

## Execution Schedule

| Day | Experiments | GPU Hours | Checkpoints |
|-----|------------|-----------|-------------|
| 1 | 2A + 2B (parallel if 2 GPUs, else sequential) | 2.2h | crossover confirmed or killed |
| 1 | 3A (gated VR, 500 steps quick check) | 0.14h | gate dynamics validated |
| 2 | 3A-3D at 2000 steps (sequential) | 4.4h | best variant identified |
| 2 | Best variant at LR=0.06 if not already tested | 0.55h | LR interaction confirmed |
| 3 | 4A: winner at 5000 steps | 1.4h | mid-range validated |
| 3 | 4C: winner + 12 layers at 5000 steps | 1.4h | depth scaling validated |
| 4 | 4B: winner at 10,000 steps | 2.8h | near-full-range validated |
| 5 | 5A: winner at 13,780 steps | 3.8h | final validation |

**Total GPU time**: ~16h on 1xL40S (spread over 5 days)

## Decision Gates

After each phase, stop and evaluate:

1. **After Phase 2**: If VR does NOT cross baseline by step 2000, the mechanism is fundamentally limited in this architecture. Move directly to Phase 3 weighted variants.

2. **After Phase 3**: Pick the variant with the best BPB at step 2000. Must beat baseline by >= 0.005 BPB to justify continuing.

3. **After Phase 4A (5000 steps)**: Must beat baseline by >= 0.003 BPB (improvement can narrow somewhat at scale). If < 0.003, the improvement is likely noise.

4. **After Phase 4B (10,000 steps)**: Must beat baseline by >= 0.002 BPB. This is the go/no-go for Phase 5.

5. **After Phase 5**: Must achieve int8+zlib val_bpb <= 1.2194 for submission candidacy.

## What NOT to Test

- Cumsum variant: already shown to hurt at 200 steps, no convergence trend
- Value residual without resid_mix: resid_mix serves a different purpose (embedding injection)
- AttnRes + architecture changes simultaneously: one variable at a time until we have a winning AttnRes variant

## Implementation Notes

New variants (3A-3D) need code changes in train_gpt.py:
- 3A (gated): ~5 lines in CausalSelfAttention
- 3B (mid-layer): ~3 lines in GPT.forward
- 3C (weighted): ~20 lines in GPT (new ParameterList + weighted sum loop)
- 3D (weighted_vector): same as 3C but with dim-sized weights

All variants must add new ATTNRES_MODE string options. Keep backward compatibility with existing "none", "cumsum", "value_residual" modes.

Current line count is well under the 1500-line hard cap.
