# Knowledge Base

Persistent research memory for autonomous AI agents working on Parameter Golf.
**Update this file** whenever an experiment produces a significant finding — positive or negative.

Last updated: 2026-03-21

---

## Proven Facts (experimentally confirmed, high confidence)

### Activation Functions (60+ runs, 500–13000 steps)

1. **Squaring the activation is the single most impactful design choice.** All squared activations (relu², selu², abs², leaky²) significantly outperform their unsquared versions. The specific base function matters much less than whether you square it.

2. **Gradients must scale with activation magnitude (H1).** Flat/constant gradients cost +0.08–0.11 BPB (27–37x noise floor). This is the largest single effect measured. relu²'s backward gradient (2x for x>0) naturally provides this. Any replacement activation must preserve this property.

3. **Don't compress output range (H2).** Bounded activations (tanh², sigmoid², erf²) cost +0.01 to +0.10 BPB. The more aggressively outputs are compressed, the worse the result.

4. **Let negative signal through (H3).** leaky(0.5)² beats relu² by ~0.003 BPB consistently from 2000–6000 steps. The gap is stable and survives quantization. However, 0.003 BPB is below the competition's 0.005 significance threshold — **not enough to win on its own**.

5. **leaky(0.5)² is the best activation found.** Satisfies all three rules (H1+H2+H3). Progression: -0.010 at 500 steps → -0.003 at 2000+ steps (the early advantage is inflated by init variance).

6. **Don't square gated activations.** swiglu² costs +0.139 BPB (46x noise), swirelu² costs +0.042 BPB. The double multiplicative interaction is unstable.

7. **Quantization preserves activation ranking.** The int8 quant gap is ~0.005 BPB for all activations tested. Rankings are identical pre- and post-quant.

8. **500-step rankings are unreliable within the squared family.** abs² leads at 500 steps but drops to 3rd by 6000 steps. Any activation within ~0.01 BPB at 500 steps requires 2000+ step validation.

9. **relu² keeps improving through 13k steps** with no sign of collapse. Post-quant at 13k: 1.2498 BPB (vs leaderboard baseline 1.2244).

### Falsified Hypotheses

| Hypothesis | Test | Result |
|---|---|---|
| "Sparsity (hard zeros) is why relu² works" | abs² has no zeros, performs equally | **Falsified** |
| "Bigger outputs explain ²" | 2·relu(x) matches scale but scored +0.023 worse | **Falsified** — quadratic shape matters |
| "Gates always help" | gated_relu² tied with relu² at matched params | **Falsified** — width lost to gate costs as much as gate contributes |

### Architecture (preliminary, results still incoming)

- **Baseline model**: 9L, 512dim, 8 heads (4 KV), 1024 vocab, tied embeddings, ~17M params
- **Leaderboard baseline**: 1.2244 BPB (8xH100, 600s)
- **Our best L40S result**: relu² at 13k steps → 1.2498 post-quant BPB
- Architecture experiments (weight sharing, factorized embeddings, depthwise conv, MoE, QAT) are queued/running — results pending as of 2026-03-21

## Failed Approaches (don't retry without new evidence)

- **LR tuning** — FORBIDDEN per project rules. Focus on architecture/mechanism changes only.
- **Bounded activations** (tanh, sigmoid, erf, arctan) — all squared variants hurt due to output compression
- **Squaring gated activations** (swiglu², swirelu²) — catastrophically unstable
- **Constant-gradient activations** — +0.08-0.11 BPB penalty, violates H1
- **Pure output scaling** (2·relu vs relu²) — does not replicate the benefit of squaring

## Open Questions

- [ ] Does weight sharing (layer cycling) help? 5x2, 4x3 configs running
- [ ] Factorized embeddings — how much can we save and reinvest?
- [ ] Depthwise causal conv — free local patterns?
- [ ] Soft MoE at 17M scale — does routing overhead eat the capacity gain?
- [ ] QAT — can we close the 0.005 quant gap (or the 0.0325 gap from 4h run)?
- [ ] leaky(0.5)² at full 13k steps — does the 0.003 gap hold?
- [ ] Combination experiments — stacking multiple small wins

## Current Best Configuration

```
Architecture:  9-layer GPT, 512-dim, 8 heads, 4 KV heads
Activation:    relu² (proven) or leaky(0.5)² (marginal +0.003)
Vocab:         1024 BPE (tied embeddings)
Optimizer:     Muon
Best result:   1.2498 post-quant BPB (13k steps, L40S)
Target:        <1.2194 BPB (beat baseline by ≥0.005)
Gap to close:  ~0.025 BPB (need architecture/mechanism wins)
```

## Research Pipeline

| Stage | Steps | Time (L40S) | Purpose | Threshold to advance |
|-------|------:|-------------|---------|---------------------|
| Explore | 500 | ~28 min | Screen many ideas fast | >0.01 BPB improvement |
| Validate | 2000 | ~1.8 hr | Confirm explore winners | >0.005 BPB on 2+ seeds |
| Scale | 5000 | ~4.6 hr | Near-final signal | Consistent improvement |
| Full | 13780 | ~12.7 hr | Pre-submission | Beats 1.2244 BPB |
