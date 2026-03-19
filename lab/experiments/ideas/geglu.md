# GEGLU / SwiGLU MLP Replacement

**Tier**: 2
**Category**: Architecture (MLP activation)
**Extra params**: 0 to ~500K (depends on variant)
**Extra FLOPs**: ~same

## Core Idea

Replace the current `relu²` MLP with GEGLU or SwiGLU. The current MLP does:
```python
x = relu(x @ W_up)    # relu activation
out = x.square() @ W_down  # then square — so relu(z)² = relu(z)*relu(z)
```

**Key insight**: relu² is ALREADY a form of self-gating — it's `relu(z) * relu(z)`, which means the value and gate are the same signal. SwiGLU uses two DIFFERENT signals: `swish(z_gate) * z_up`. The question is whether decoupling gate from value helps enough to justify the cost.

SwiGLU with separate projections:
```
out = (swish(x @ W_gate) * (x @ W_up)) @ W_down  # two projections
```

## Variants

1. **True SwiGLU (2/3 rule)**: Separate gate/up projections, hidden=682 to keep params equal. The only clean comparison.
2. **GEGLU**: Same structure but GELU instead of swish
3. **swish²**: `swish(xW)²` — drop-in replacement for relu², same param count, smoother gradients
4. **Keep relu² but add gate**: `relu(xW_up)² * sigmoid(xW_gate)` — additive gate on top

## Param Budget

- swish² drop-in: 0 extra params (just change activation)
- True SwiGLU (2/3 rule): hidden=682 instead of 1024 → similar param count
- With separate gate at full width: +512×1024 = 524K params/layer (too expensive)

## Risks

- relu² is not standard but was chosen by the baseline authors (from modded-nanogpt) — may be specifically optimized for this setting
- Changing MLP activation may require LR retuning
- The gains from swapping one self-gated activation for another may be very small
- True SwiGLU with 2/3 hidden dim changes the MLP shape, which interacts with Muon optimizer

---

## Literature Review Prompt

> **Prompt for literature review agent:**
>
> I have a small GPT (17M params) using relu² activation (`relu(x) * relu(x)`, a self-gated activation) in its MLP layers (hidden=2x model dim). I'm considering alternatives: SwiGLU, GEGLU, swish², or other gated activations.
>
> Research:
> 1. SwiGLU vs GEGLU vs relu² vs swish² — which performs best for small language models? relu² is self-gated (same signal gates itself), SwiGLU decouples gate from value.
> 2. The "2/3 rule" for GLU variants (reducing hidden dim to compensate for gate projection) — does this actually maintain parameter parity AND performance parity?
> 3. Has anyone compared relu² specifically against GLU variants? relu² was used in modded-nanogpt / nanoGPT speedrun work.
> 4. Does decoupling the gate signal from the value signal (as SwiGLU does vs relu²) provide meaningful benefit at small scale?
> 5. Interaction with Muon optimizer (Newton-Schulz orthogonalization on weight matrices) — does the activation function matter?
> 6. At 17M params / 9 layers, how much does MLP activation choice actually matter vs. attention design or other architecture choices?
>
> Concrete perplexity numbers preferred. Small model results most relevant.

## Literature Review

> _Paste into `/lab/research_papers/geglu.md`_

**Agent chat URL**: <!-- paste URL here -->

---

## Status

- [ ] Literature review
- [ ] Experiment: swish² drop-in (zero param cost)
- [ ] Experiment: true SwiGLU (2/3 hidden=682)
- [ ] Result analysis
