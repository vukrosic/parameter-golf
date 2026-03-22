# Tiered Architecture Screen — 2026-03-22

Ladder: 1s → 2s → 3s | Counts: 10 → top 5 → top 3

---

### Stage 1 — 1s

| Run | Duration | Architecture change | Loss | Baseline loss | Delta loss | Decision |
|---|---:|---|---:|---:|---:|---|
| tiered_1s_swiglu | 1s | SwiGLU activation | 6.9280 | 6.9312 | -0.0032 | promote ✓ |
| tiered_1s_attnres_vr | 1s | value residual attn | 6.9296 | 6.9312 | -0.0016 | promote ✓ |
| tiered_1s_moe2 | 1s | soft MoE 2 experts | 6.9304 | 6.9312 | -0.0008 | promote ✓ |
| tiered_1s_heads4 | 1s | 4 heads / 2 kv-heads | 6.9306 | 6.9312 | -0.0006 | promote ✓ |
| tiered_1s_baseline | 1s | baseline | 6.9312 | 6.9312 | +0.0000 | baseline |
| tiered_1s_untied | 1s | untied embeddings | 6.9315 | 6.9312 | +0.0003 | drop |
| tiered_1s_moe4 | 1s | soft MoE 4 experts | 6.9317 | 6.9312 | +0.0005 | drop |
| tiered_1s_conv3 | 1s | depthwise conv k=3 | 6.9320 | 6.9312 | +0.0008 | drop |
| tiered_1s_deep2 | 1s | 2 layers | 6.9322 | 6.9312 | +0.0010 | drop |
| tiered_1s_attnres_w | 1s | weighted attn residual | 6.9328 | 6.9312 | +0.0016 | drop |

---

### Stage 2 — 2s

| Run | Duration | Architecture change | Loss | Baseline loss | Delta loss | Decision |
|---|---:|---|---:|---:|---:|---|
| tiered_2s_attnres_vr | 2s | value residual attn | 6.9725 | 6.9812 | -0.0087 | promote ✓ |
| tiered_2s_swiglu | 2s | SwiGLU activation | 6.9763 | 6.9812 | -0.0049 | promote ✓ |
| tiered_2s_moe2 | 2s | soft MoE 2 experts | 6.9796 | 6.9812 | -0.0016 | promote ✓ |
| tiered_2s_baseline | 2s | baseline | 6.9812 | 6.9812 | +0.0000 | baseline |
| tiered_2s_heads4 | 2s | 4 heads / 2 kv-heads | 6.9912 | 6.9812 | +0.0101 | drop |

---

### Stage 3 — 3s

| Run | Duration | Architecture change | Loss | Baseline loss | Delta loss | Decision |
|---|---:|---|---:|---:|---:|---|
| tiered_3s_swiglu | 3s | SwiGLU activation | 6.9535 | 6.9844 | -0.0309 | finalist ✓ |
| tiered_3s_moe2 | 3s | soft MoE 2 experts | 6.9568 | 6.9844 | -0.0276 | finalist ✓ |
| tiered_3s_attnres_vr | 3s | value residual attn | 6.9577 | 6.9844 | -0.0267 | finalist ✓ |
| tiered_3s_baseline | 3s | baseline | 6.9844 | 6.9844 | +0.0000 | baseline |

---

## Conclusion

SwiGLU, soft MoE-2, and value residual attention all beat the baseline consistently across every stage. heads4 looked promising at 1s but flipped negative at 2s — unreliable at this scale. The 1s ranking was a strong predictor of the 3s outcome, meaning the ladder is stable and these three are worth promoting to a proper validation run (500+ steps).
