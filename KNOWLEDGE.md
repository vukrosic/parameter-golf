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

### Architecture (71 experiments completed, 500-step screens)

- **Baseline model**: 9L, 512dim, 8 heads (4 KV), 1024 vocab, tied embeddings, ~17M params → 12.6 MB submission
- **Submission limit**: 16 MB (int8 zlib-compressed). All architecture choices must respect this.
- **Leaderboard baseline**: 1.2244 BPB (8xH100, 600s)
- **Our best L40S result**: relu² at 13k steps → 1.2498 post-quant BPB

10. **MoE 4-expert is powerful but exceeds 16 MB budget.** At dim=512, 4 experts → 19.6 MB (22% over). The -0.048 BPB gain vs baseline is confounded by having 55% more parameters. Need reduced-dim experiments to determine if the architecture wins at iso-parameter budget.

11. **MoE expert count matters most.** 4 > 3 >> 2 experts. The jump from 2→4 is worth ~0.04 BPB at 500 steps. But only 2-expert fits under 16 MB at dim=512 (-0.006 to -0.019 BPB).

12. **Untied factored embeddings (bn128) are the best legal architecture change.** -0.031 BPB at 12.6 MB. Tied factored embeddings are catastrophic — bn128 tied *hurts* by +0.076 BPB. The untied/tied distinction is the critical factor, not the bottleneck width.

13. **Depthwise convolution is a dead end.** Every variant hurts (+0.025 to +0.174 BPB). Monotonically worse with larger kernels. Combinations with other techniques compound damage.

14. **Weight sharing is not competitive alone** but produces small models (10.8 MB). Only ws_5x2_wide_leaky beats baseline (-0.010 BPB). Potentially useful combined with other techniques that add parameters.

15. **QAT alone is within noise (+/-0.003 BPB).** Does not hurt MoE — important for final submission.

## Failed Approaches (don't retry without new evidence)

- **LR tuning** — FORBIDDEN per project rules. Focus on architecture/mechanism changes only.
- **Bounded activations** (tanh, sigmoid, erf, arctan) — all squared variants hurt due to output compression
- **Squaring gated activations** (swiglu², swirelu²) — catastrophically unstable
- **Constant-gradient activations** — +0.08-0.11 BPB penalty, violates H1
- **Pure output scaling** (2·relu vs relu²) — does not replicate the benefit of squaring
- **Depthwise convolution** — all kernel sizes hurt (+0.025 to +0.174 BPB), bigger = worse
- **Tied factored embeddings** — catastrophic at all bottleneck sizes (+0.058 to +0.379 BPB)
- **QAT alone** — within noise, no measurable benefit without other changes
- **Conv + anything combos** — conv damage compounds with other techniques

## Open Questions

- [x] ~~Does weight sharing help?~~ Only ws_5x2_wide_leaky beats baseline (-0.010). Not competitive alone.
- [x] ~~Factorized embeddings?~~ Untied bn128 = -0.031 BPB (best legal result). Tied = catastrophic.
- [x] ~~Depthwise causal conv?~~ Dead end. All variants hurt.
- [x] ~~Soft MoE at 17M scale?~~ Powerful (+0.048) but 4-expert blows 16 MB budget by 22%.
- [x] ~~QAT alone?~~ Within noise. Doesn't hurt MoE — useful for submission.
- [ ] leaky(0.5)² at full 13k steps — does the 0.003 gap hold?
- [ ] **Can MoE 4-expert fit under 16 MB at reduced dim (~448)?** Highest priority.
- [ ] **Does MoE 2e + untied bn128 + leaky stack?** Best legal combo, untested.
- [ ] **Does untied bn128 hold at 2000+ steps with multiple seeds?**

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

### Best Legal Architecture Candidates (500-step, pending validation)

```
#1  Untied bn128 embeddings:     1.4483 BPB (-0.031), 12.6 MB
#2  MoE 2e + 11L:                1.4606 BPB (-0.019), 15.4 MB
#3  MoE 2e + leaky:              1.4698 BPB (-0.010), 12.7 MB
    MoE 2e + untied + leaky:     UNTESTED (estimated ~12.8 MB)
    MoE 4e at dim=448:           UNTESTED (estimated ~15 MB)
```

## Research Pipeline

| Stage | Steps | Time (L40S) | Purpose | Threshold to advance |
|-------|------:|-------------|---------|---------------------|
| Explore | 500 | ~28 min | Screen many ideas fast | >0.01 BPB improvement |
| Validate | 2000 | ~1.8 hr | Confirm explore winners | >0.005 BPB on 2+ seeds |
| Scale | 5000 | ~4.6 hr | Near-final signal | Consistent improvement |
| Full | 13780 | ~12.7 hr | Pre-submission | Beats 1.2244 BPB |
