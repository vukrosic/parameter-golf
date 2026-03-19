# AttnRes Experiment Plan

**Paper**: Attention Residuals — Inter-layer weighted attention replacing standard residual connections
**Research summary**: `/lab/research_papers/attnres.md`

## What the Baseline Already Has

The current `train_gpt.py` already implements primitive cross-layer connections:
1. **`resid_mix`** (Block L637-641): Each block mixes `x` with `x0` (embedding output) via learnable weights
2. **`skip_weights`** (GPT L673, 706-713): U-Net encoder-decoder with weighted skip connections
3. **`attn_scale` / `mlp_scale`** (Block L635-636): Per-layer scaling on attn/mlp outputs

These are good but limited:
- `resid_mix` only connects to the embedding, not intermediate layers
- `skip_weights` only connects encoder→decoder (first half → second half)
- No layer can attend to ALL previous layers' representations

## What AttnRes Adds

The key idea: **each layer should be able to attend over ALL previous layer outputs**, not just the embedding or its U-Net skip partner.

For a 9-layer model, this means layer 8 can directly access information from layers 0-7 via learned attention weights, rather than depending on the residual stream to propagate everything sequentially.

## Experiment Variants (Ordered by Risk/Complexity)

### Experiment A: `attnres_cumsum` — Cumulative Attention Output Residual
**Simplest. Tests the core hypothesis with zero new parameters.**

Modification: Maintain a running sum of all previous attention outputs and add it to each block's input.

```python
# In GPT.forward():
attn_residual = torch.zeros_like(x)
for i, block in enumerate(self.blocks):
    x, attn_out = block(x, x0, attn_residual)  # block returns attn_out too
    attn_residual = attn_residual + attn_out
```

- Extra params: **0**
- Extra VRAM: 1 tensor (batch, seq, dim) for the running sum
- Extra compute: 1 addition per layer
- Risk: Very low — if it hurts, the residual sum was noise

### Experiment B: `attnres_value_residual` — Cross-Layer Value Residual
**From paper variant [3]. Passes first-layer values to all subsequent layers.**

Modification: Store value embeddings from layer 0, add them to value inputs of all subsequent layers.

```python
# In CausalSelfAttention.forward(), accept optional v_residual:
v = self.c_v(x) + v_residual  # if v_residual is provided
```

- Extra params: **0** (or 1 learnable scalar gate per layer = 9 params)
- Extra VRAM: 1 tensor for stored values from layer 0
- Extra compute: negligible
- Claim: Reduces attention concentration in deeper layers

### Experiment C: `attnres_weighted` — Learned Inter-Layer Weights
**From our original plan. Each layer learns a softmax-weighted combination of all previous outputs.**

```python
# New: per-layer weight vectors
self.layer_weights = nn.ParameterList([
    nn.Parameter(torch.zeros(i + 1))  # one weight per previous layer + embedding
    for i in range(num_layers)
])

# In GPT.forward():
layer_outputs = [x]  # start with embedding
for i, block in enumerate(self.blocks):
    weights = F.softmax(self.layer_weights[i], dim=0)
    x_input = sum(w * out for w, out in zip(weights, layer_outputs))
    x = block.forward_no_mix(x_input)  # skip the resid_mix since we replaced it
    layer_outputs.append(x)
```

- Extra params: sum(1..9) = **45 scalars** (negligible)
- Extra VRAM: Store all 9 layer outputs (~9 * batch * seq * 512)
- Extra compute: Weighted sum per layer
- This is the most principled variant — replaces `resid_mix` and `skip_weights` entirely

### Experiment D: `attnres_weighted_vector` — Per-Dimension Weights
**Like C but with vector weights (dim=512) instead of scalar per layer.**

```python
self.layer_weights = nn.ParameterList([
    nn.Parameter(torch.zeros(i + 1, model_dim))
    for i in range(num_layers)
])
# softmax over layers dim, applied per-dimension
```

- Extra params: sum(1..9) * 512 = **23,040** (~23KB int8, fits easily)
- Allows different dimensions to route from different layers
- Most expressive variant

## Integration with Current LR Search

Phase 1 LR sweep is running now. AttnRes experiments should:
1. **Use best LR from Phase 1** (currently MATRIX_LR=0.06 is leading)
2. **Run at 500 steps** for initial comparison, extend winners to 1000
3. **Compare against baseline at same step count** with same LR

Experiment order:
| Priority | ID | Steps | Description |
|----------|-----|-------|-------------|
| 1 | `attnres_cumsum` | 500 | Zero-param cumulative attention residual |
| 2 | `attnres_value_residual` | 500 | Cross-layer value residual from layer 0 |
| 3 | `attnres_weighted` | 500 | Scalar softmax weights over all prev layers |
| 4 | `attnres_weighted_vector` | 500 | Per-dim vector weights over all prev layers |
| 5 | Winner | 1000 | Extend best variant |
| 6 | Winner + 12 layers | 1000 | Test if AttnRes enables deeper models |

All experiments use: MATRIX_LR=0.06 (or best from Phase 1), VAL_LOSS_EVERY=100

## Implementation Strategy

### What to modify in `train_gpt.py`:

**For variants A & B**: Minimal changes
- Modify `Block.forward()` to return attention output separately
- Modify `GPT.forward()` to accumulate and pass residuals
- ~20 lines of code

**For variants C & D**: Replace residual routing
- Add `layer_weights` parameter to GPT
- Modify `GPT.forward()` to store layer outputs and compute weighted sums
- Can remove `resid_mix` and `skip_weights` (replaced by more general mechanism)
- ~40 lines of code

### What NOT to change:
- Attention mechanism itself (Q/K/V projections, RoPE, etc.)
- MLP structure
- Normalization (already Pre-LN via RMSNorm — matches paper recommendation)
- Quantization (AttnRes params are tiny, kept as fp16 passthrough)

## Size Budget

| Component | Bytes (int8+zlib) |
|-----------|-------------------|
| Baseline model | ~15.86 MB |
| AttnRes A (cumsum) | +0 bytes |
| AttnRes B (value res) | +0 bytes (or +36 bytes for gates) |
| AttnRes C (scalar weights) | +90 bytes |
| AttnRes D (vector weights) | ~+23 KB |
| **Headroom remaining** | **~140 KB** |

All variants fit easily within the 16MB cap.

## Success Criteria

- Any variant that beats baseline by >= 0.003 BPB at 500 steps → extend to 1000
- Any variant that beats baseline by >= 0.005 BPB at 1000 steps → candidate for final submission
- Target: Combined with LR tuning, achieve <= 1.2194 BPB (0.005 below current record)

## Status: READY TO IMPLEMENT
Paper read. Code analyzed. Variants designed. Waiting for Phase 1 LR results to set base LR.
