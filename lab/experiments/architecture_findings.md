# Architecture Experiments: Pushing the Pareto Frontier of Parameter-Constrained Language Models

## Overview

This document records systematic architecture experiments for the [Parameter Golf](https://github.com/openai/parameter-golf) challenge — training the best language model that fits in 16MB compressed and trains in under 10 minutes on 8×H100s.

**Baseline model**: 9-layer GPT, 512-dim, 8 heads (4 KV), 1024 vocab, tied embeddings, ReLU² activation, ~17M params.
**Evaluation metric**: Bits-per-byte (BPB) on FineWeb validation set, lower is better.
**Leaderboard baseline**: 1.2244 BPB.
**Target**: Beat baseline by ≥0.005 BPB.

All experiments run at **500 steps** on single RTX 5090 GPUs for rapid screening. Promising findings are validated at 2000+ steps.

---

## Experiment 1: Weight Sharing / Layer Cycling

**Hypothesis**: A parameter-constrained model benefits more from depth than width. By reusing a small set of unique transformer blocks multiple times, we can increase effective depth without adding parameters. Saved parameters can be reinvested into wider layers.

### Setup

| Config | Unique Blocks | Cycles | Effective Depth | Model Dim | Extra Notes |
|--------|:---:|:---:|:---:|:---:|---|
| baseline | 9 | 1 | 9 | 512 | Standard config |
| ws_5x2 | 5 | 2 | 10 | 512 | Basic cycling |
| ws_5x2_wide | 5 | 2 | 10 | 640 | Reinvest params |
| ws_4x3 | 4 | 3 | 12 | 512 | Aggressive sharing |
| ws_3x3 | 3 | 3 | 9 | 512 | Same depth, fewer params |
| ws_9x2 | 9 | 2 | 18 | 512 | Pure depth, no savings |

### Results

| Config | Val BPB (500 steps) | Post-Quant BPB | Params | Notes |
|--------|:---:|:---:|:---:|---|
| baseline | — | — | — | Running |
| ws_5x2 | — | — | — | Running |
| ws_5x2_wide | — | — | — | Running |
| ws_4x3 | — | — | — | Running |

### Key Findings

> *Results pending — will update as experiments complete.*

---

## Experiment 2: Factorized Embeddings

**Hypothesis**: With only 1024 BPE tokens, the full 512-dim embedding is wasteful. A low-rank bottleneck (vocab → 64 → 512) saves ~400KB that can be reinvested into model capacity.

### Setup

| Config | Bottleneck Dim | Model Dim | Layers | Extra Notes |
|--------|:---:|:---:|:---:|---|
| baseline | 512 (full) | 512 | 9 | Standard |
| bn32 | 32 | 512 | 9 | Aggressive |
| bn64 | 64 | 512 | 9 | Balanced |
| bn128 | 128 | 512 | 9 | Conservative |
| bn64_w576 | 64 | 576 | 9 | Reinvest → width |
| bn64_10L | 64 | 512 | 10 | Reinvest → depth |

### Results

| Config | Val BPB (500 steps) | Post-Quant BPB | Params | Notes |
|--------|:---:|:---:|:---:|---|
| baseline | — | — | — | Running |
| bn64 | — | — | — | Running |

### Key Findings

> *Results pending.*

---

## Experiment 3: Depthwise Causal Convolution

**Hypothesis**: Adding a cheap causal depthwise conv1d before attention in each layer gives the model a local receptive field for free (~1.5K params/layer). This lets attention focus on long-range dependencies instead of wasting heads on adjacent-token patterns.

### Setup

| Config | Kernel Size | Extra Params/Layer | Total Extra | Notes |
|--------|:---:|:---:|:---:|---|
| baseline | none | 0 | 0 | No conv |
| k3 | 3 | 1,536 | 13,824 | Trigram window |
| k5 | 5 | 2,560 | 23,040 | 5-gram window |
| k7 | 7 | 3,584 | 32,256 | 7-gram window |
| k11 | 11 | 5,632 | 50,688 | Large window |

### Results

| Config | Val BPB (500 steps) | Post-Quant BPB | Step Time | Notes |
|--------|:---:|:---:|:---:|---|
| baseline | — | — | — | Running |
| k3 | — | — | — | Running |
| k5 | — | — | — | Running |

### Key Findings

> *Results pending.*

---

## Experiment 4: Soft Mixture of Experts (MoE)

**Hypothesis**: Replacing the single MLP with N half-width expert MLPs (soft-merged via a learned router) doubles MLP capacity at the same parameter count. Soft merging avoids discrete routing issues and is trivially efficient.

### Setup

| Config | Experts | Expert Width | Router | Total MLP Params |
|--------|:---:|:---:|:---:|---|
| baseline | 1 | 2× model_dim | none | ~same |
| 2e | 2 | 1× model_dim | softmax(Wx) | ~same |
| 4e | 4 | 0.5× model_dim | softmax(Wx) | ~same |

### Results

| Config | Val BPB (500 steps) | Post-Quant BPB | Step Time | Notes |
|--------|:---:|:---:|:---:|---|
| baseline | — | — | — | Running |
| 2e | — | — | — | Running |
| 4e | — | — | — | Running |

### Key Findings

> *Results pending.*

---

## Experiment 5: Quantization-Aware Training (QAT)

**Hypothesis**: The int8 post-training quantization gap (0.0325 BPB on 4h run) represents the single largest unaddressed loss. Training with simulated quantization noise (straight-through estimator) makes weights more robust to quantization.

### Setup

| Config | QAT Start | Fake Quant Method | Notes |
|--------|:---:|:---:|---|
| baseline | never | none | Standard training |
| qat_90pct | 90% of steps | per-row int8 STE | Conservative |
| qat_70pct | 70% of steps | per-row int8 STE | Balanced |
| qat_50pct | 50% of steps | per-row int8 STE | Moderate |
| qat_30pct | 30% of steps | per-row int8 STE | Aggressive |

### Results

| Config | Val BPB (pre-quant) | Val BPB (post-quant) | Quant Gap | Notes |
|--------|:---:|:---:|:---:|---|
| baseline | — | — | — | Running |
| qat_70pct | — | — | — | Running |

### Key Findings

> *Results pending.*

---

## Combination Experiments

After identifying winners from individual experiments, test combinations:

| Config | Architecture Changes | Val BPB | Notes |
|--------|---|:---:|---|
| best_combo_1 | TBD | — | Based on individual winners |
| best_combo_2 | TBD | — | — |
| best_combo_3 | TBD | — | — |

---

## Methodology Notes

1. **500-step screening**: Fast but noisy. Results with < 0.01 BPB difference from baseline are within noise. Only differences > 0.01 BPB are considered significant at 500 steps.
2. **Validation**: Promising results (> 0.005 BPB improvement) will be validated at 2000+ steps on multiple seeds.
3. **Parameter accounting**: All experiments account for the 16MB compressed model size constraint.
4. **Step time**: Reported to assess feasibility within the 600s wallclock budget on 8×H100.
5. **Hardware**: All screening experiments on single RTX 5090 (32GB VRAM).

---

## Summary of Findings

> *Will be populated as experiments complete. Expected completion: 24-48 hours.*

### What Worked

*(pending)*

### What Didn't Work

*(pending)*

### Surprising Results

*(pending)*

### Recommended Configuration

*(pending)*

---

*Generated: 2026-03-21*
*Challenge: [OpenAI Parameter Golf](https://github.com/openai/parameter-golf)*
