# AttnRes Experiment Plan

**Paper**: Attention Residuals — Inter-layer weighted attention replacing standard residual connections
**Claim**: ~25% improvement for <5% overhead

## Why This Fits Parameter Golf

The 16MB constraint means we can't just scale up. AttnRes adds minimal parameters
(a few learned weight vectors per layer) but routes information more effectively between layers.
For a 9-layer model, this is potentially huge — better inter-layer information flow
without adding model dimensions.

## Architecture Variants to Test

### Variant 1: Full AttnRes (Small Model Friendly)
- Static query vectors per layer (learnable `w_t` per layer)
- Keys/Values = RMSNorm(layer_outputs)
- Softmax attention across all previous layer outputs
- Parameter cost: ~9 * 512 = 4,608 extra params (negligible)

### Variant 2: Block AttnRes
- Isolate embedding layer as its own block
- Group remaining 8 layers into ~2-4 blocks
- Sum within blocks, attend between blocks
- Even cheaper but may lose fine-grained routing

### Variant 3: Simplified — Learnable Residual Weights
- Minimum viable version: learnable scalar weights per skip connection
- No attention mechanism, just `x_t = sum(a_{t,s} * y_s)` with learned `a`
- Cheapest test of the core hypothesis

## Experiment Sequence

All on 1xL40S, 500 steps first, extend winners to 1000+.

| ID | Config | Description |
|----|--------|-------------|
| `attnres_v1_full` | Full AttnRes | Static queries, per-layer attention over all prev outputs |
| `attnres_v2_block` | Block AttnRes | 2-4 blocks, compressed states |
| `attnres_v3_scalar` | Scalar weights | Learnable skip weights, no attention |
| `attnres_v1_deeper` | Full AttnRes + 12 layers | Test if AttnRes enables going deeper |

## Implementation Notes

- Must modify the transformer forward pass in train_gpt.py
- Need to store all intermediate layer outputs (VRAM cost: 9 * batch * seq * 512)
- RMSNorm already exists in the codebase — reuse it
- Attention weights must be non-negative (softmax enforces this)
- Sum-to-1 constraint: safe because RMSNorm makes model scale-invariant

## Size Budget Check
- Baseline: ~15.86 MB (int8+zlib)
- AttnRes v1 adds: ~4,608 params * 1 byte (int8) = ~4.5 KB
- AttnRes v2 adds: even less
- Headroom: ~140 KB — plenty of room

## Status: AWAITING PAPER PASTE
Need full paper content in `/lab/research_papers/attnres.md` before implementing.
