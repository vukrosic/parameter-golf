# Architecture Search Findings (500-step Explorations)

**Date:** 2026-03-21
**Experiments:** 71 architecture variants at 500 steps each
**Baselines:** arch_embed_baseline (1.4793), arch_moe_baseline (1.4794), arch_conv_baseline (1.4833)

---

## Ranked Results (all 71 experiments)

| Rank | Experiment | val_bpb | Delta vs baseline | Verdict |
|------|-----------|---------|-------------------|---------|
| 1 | arch_moe_4e_leaky | 1.4317 | **-0.048** | **WINNER** |
| 2 | arch_moe_4e | 1.4373 | -0.042 | Promising |
| 3 | arch_moe4_qat70 | 1.4377 | -0.042 | Promising |
| 4 | arch_moe_4e_mlp4x | 1.4381 | -0.041 | Promising |
| 5 | arch_embed_bn128_untied | 1.4483 | -0.031 | Promising |
| 6 | arch_moe_3e | 1.4533 | -0.026 | Promising |
| 7 | arch_embed_bn64_untied | 1.4557 | -0.024 | Promising |
| 8 | arch_moe_2e_11L | 1.4606 | -0.019 | Moderate |
| 9 | arch_moe_2e_10L | 1.4659 | -0.014 | Moderate |
| 10 | arch_ws_5x2_wide_leaky | 1.4689 | -0.010 | Moderate |
| 11 | arch_moe_2e_leaky | 1.4698 | -0.010 | Moderate |
| 12 | arch_moe2_qat70 | 1.4715 | -0.008 | Moderate |
| 13 | arch_qat_70_leaky | 1.4722 | -0.007 | Moderate |
| 14 | arch_ws_9x2 | 1.4727 | -0.007 | Moderate |
| 15 | arch_qat_50_leaky | 1.4728 | -0.007 | Moderate |
| 16 | arch_moe_2e | 1.4733 | -0.006 | Marginal |
| 17 | arch_moe_2e_mlp3x | 1.4740 | -0.005 | Marginal |
| 18 | arch_qat_90pct | 1.4774 | -0.002 | Marginal |
| 19 | arch_ws_5x2_wide | 1.4789 | -0.000 | Neutral |
| 20 | **arch_embed_baseline** | **1.4793** | **0.000** | **BASELINE** |
| 21 | **arch_moe_baseline** | **1.4794** | **0.000** | **BASELINE** |
| 22 | arch_qat_70pct | 1.4810 | +0.002 | Neutral |
| 23 | arch_qat_50pct | 1.4812 | +0.002 | Neutral |
| 24 | arch_qat_from_start | 1.4815 | +0.002 | Neutral |
| 25 | arch_qat_30pct | 1.4820 | +0.003 | Neutral |
| 26 | **arch_conv_baseline** | **1.4833** | **+0.004** | **BASELINE** |
| 27 | arch_baseline_9L | 1.4847 | +0.005 | Neutral |
| 28 | arch_ws_4x3_wide | 1.4868 | +0.007 | Slight hurt |
| 29 | arch_ws_6x2 | 1.4886 | +0.009 | Slight hurt |
| 30-71 | *(remaining 42 experiments)* | 1.50-1.89 | +0.02 to +0.41 | Bad to Terrible |

---

## Analysis by Architecture Category

### 1. Mixture of Experts (MOE) — CLEAR WINNER

**Best:** 1.4317 (4 experts + leaky) | **Worst:** 1.4740 (2 experts + mlp3x)

MOE is the single most impactful architecture change tested. Every MOE variant beats baseline.

| Variant | val_bpb | Delta | Notes |
|---------|---------|-------|-------|
| moe_4e_leaky | 1.4317 | -0.048 | Best overall. Leaky ReLU + 4 experts = sweet spot |
| moe_4e | 1.4373 | -0.042 | 4 experts without activation change still very strong |
| moe4_qat70 | 1.4377 | -0.042 | QAT adds nothing on top of 4-expert MOE |
| moe_4e_mlp4x | 1.4381 | -0.041 | Wider MLP doesn't help over standard 4e |
| moe_3e | 1.4533 | -0.026 | 3 experts solid but clearly worse than 4 |
| moe_2e_11L | 1.4606 | -0.019 | 2 experts + extra depth partially compensates |
| moe_2e_10L | 1.4659 | -0.014 | 2 experts + 10 layers |
| moe_2e_leaky | 1.4698 | -0.010 | Leaky helps less with fewer experts |
| moe2_qat70 | 1.4715 | -0.008 | QAT marginal on 2-expert |
| moe_2e | 1.4733 | -0.006 | 2 experts alone barely above noise |
| moe_2e_mlp3x | 1.4740 | -0.005 | Wider MLP doesn't rescue 2-expert |

**Key insight:** Expert count matters most. 4 > 3 >> 2. The jump from 2→4 experts is worth ~0.04 BPB. Leaky ReLU adds ~0.005 on top. QAT and MLP width don't help once you have 4 experts.

---

### 2. Factored Embeddings — UNTIED WORKS, TIED IS A DISASTER

**Best:** 1.4483 (bn128 untied) | **Worst:** 1.8863 (bn16 w576)

| Variant | val_bpb | Delta | Notes |
|---------|---------|-------|-------|
| embed_bn128_untied | 1.4483 | -0.031 | **Strong.** Separate input/output + 128-dim bottleneck |
| embed_bn64_untied | 1.4557 | -0.024 | Untied + 64 also good |
| embed_bn256 | 1.5375 | +0.058 | Tied — hurts |
| embed_bn128_leaky | 1.5459 | +0.067 | Tied + leaky — still bad |
| embed_bn128 | 1.5555 | +0.076 | Tied — bad |
| embed_bn64 | 1.6100 | +0.131 | Tied — very bad |
| embed_bn32 | 1.7143 | +0.235 | Tied — terrible |
| embed_bn16 | 1.8579 | +0.379 | Tied — catastrophic |

**Key insight:** Untied vs tied is the critical factor. Untied bn128 beats baseline by 0.031; tied bn128 *hurts* by 0.076. Smaller bottlenecks with tied embeddings are progressively catastrophic. Width adjustments (w544, w576) don't rescue tied variants.

---

### 3. Depthwise Convolution — DEAD END

**Best:** 1.5045 (k3 + 11L) | **Worst:** 1.6532 (k5 + bn64)

Every single convolution experiment hurts. Bigger kernels = worse results. Combinations with other techniques compound damage.

| Variant | val_bpb | Delta |
|---------|---------|-------|
| conv_k3_11L | 1.5045 | +0.025 |
| conv_k3 | 1.5134 | +0.034 |
| conv_k5_leaky | 1.5316 | +0.052 |
| conv_k5 | 1.5384 | +0.059 |
| conv_k7 | 1.5606 | +0.081 |
| conv_k9 | 1.5707 | +0.091 |
| conv_k11 | 1.5877 | +0.108 |
| conv_k15 | 1.5911 | +0.112 |
| conv_k5_ws5x2 | 1.5831 | +0.104 |
| conv_k5_bn64 | 1.6532 | +0.174 |

**Verdict:** Abandon. Standard attention is strictly superior for this task/scale.

---

### 4. Weight Sharing — PARAMETER-EFFICIENT BUT COSTLY

**Best:** 1.4689 (5x2 wide leaky) | **Worst:** 1.5866 (6x2 wide576)

| Variant | val_bpb | Delta | Params | Notes |
|---------|---------|-------|--------|-------|
| ws_5x2_wide_leaky | 1.4689 | -0.010 | 15.0M | Only WS variant solidly beating baseline |
| ws_9x2 | 1.4727 | -0.007 | 17.1M | 9 unique blocks = basically full model |
| ws_5x2_wide | 1.4789 | -0.000 | 15.0M | Matches baseline |
| ws_4x3_wide | 1.4868 | +0.007 | ~12M | |
| ws_6x2 | 1.4886 | +0.009 | 11.5M | |
| ws_5x2 | 1.5102 | +0.031 | 9.1M | Not wide enough |
| ws_3x3 | 1.5465 | +0.067 | 6.0M | Too few unique blocks |

**Verdict:** Not competitive with MOE. Only useful if parameter budget is a hard constraint.

---

### 5. QAT — NEARLY NEUTRAL ALONE

| Variant | val_bpb | Delta | Notes |
|---------|---------|-------|-------|
| qat_70_leaky | 1.4722 | -0.007 | Leaky doing the work, not QAT |
| qat_50_leaky | 1.4728 | -0.007 | Same story |
| qat_90pct | 1.4774 | -0.002 | |
| qat_70pct | 1.4810 | +0.002 | ~neutral |
| qat_50pct | 1.4812 | +0.002 | |
| qat_from_start | 1.4815 | +0.002 | |
| qat_30pct | 1.4820 | +0.003 | |

**Key insight:** QAT alone is within noise (±0.003). The leaky ReLU is doing all the work in "qat + leaky" variants. QAT start timing barely matters. However, moe4_qat70 (1.4377) matches plain moe_4e (1.4373) — QAT doesn't hurt MOE, important for final submission.

---

## What Works vs What Doesn't

### Pursue to Validation
1. **4-expert MOE + leaky ReLU** — -0.048 BPB, clear winner
2. **Untied factored embeddings (bn128)** — -0.031 BPB, worth combining with MOE
3. **MOE + QAT combo** — preserves quantized quality for submission

### Dead Ends (Do Not Revisit)
1. **Depthwise convolution** — every variant hurts, bigger kernels hurt more
2. **Tied factored embeddings** — catastrophic, especially at small bottlenecks
3. **Weight sharing** — not competitive with MOE for quality
4. **QAT alone** — nearly zero impact
5. **Conv + anything combos** — compounds the damage

### Next Steps
1. Validate MOE 4e + leaky at 2000-5000 steps, multiple seeds
2. Test MOE 4e + leaky + untied bn128 — the two biggest wins may stack
3. Full run (13780 steps) of best combo to see if it beats 1.2244 BPB leaderboard

---

*71 experiments, 3 GPUs, ~500 steps each. Generated 2026-03-21.*
