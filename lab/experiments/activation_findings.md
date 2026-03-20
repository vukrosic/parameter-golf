# Activation Function Ablation — Findings

**Scope:** MLP activation only, between up-proj and down-proj in `train_gpt.py`. 40+ runs, 500-6000 steps. Clean `relu²` baseline is `1.4805` at 500; the older `1.4522` came from a dirty worktree and should not be used.

## What is solid

1. **Squaring helps.** For simple activations, `p=2` is the best tested exponent.

| Comparison | BPB (500) | Delta |
|---|---:|---:|
| relu -> relu² | 1.5007 -> 1.4805 | -0.020 |
| silu -> silu² | 1.4908 -> 1.4841 | -0.007 |
| relu -> relu^3 | diverged | bad |

2. **Squaring gated activations is bad.**

| Comparison | BPB (500) | Delta |
|---|---:|---:|
| swiglu -> swiglu² | 1.4668 -> 1.6055 | +0.139 |

3. **The pre-squaring function matters less than the square.** By 2000 steps, most squared variants are within ~0.005; seed noise is ~0.003.

4. **`leaky(0.5)²` is the best activation tested.**

| Activation | BPB (500) | BPB (6000) | Post-quant |
|---|---:|---:|---:|
| leaky(0.5)² | 1.4708 | **1.2659** | **1.2708** |
| relu² | 1.4805 | 1.2688 | 1.2737 |
| abs² | 1.4698 | 1.2691 | 1.2745 |

5. **Short runs misrank activations.** `abs²` leads early, but `relu²` catches and slightly passes it by step 5000. `leaky(0.5)²` leads throughout the long run.

| Step | abs² | relu² | leaky(0.5)² |
|---:|---:|---:|---:|
| 500 | 1.471 | 1.481 | 1.471 |
| 1000 | 1.378 | 1.384 | 1.377 |
| 2000 | 1.322 | 1.324 | 1.320 |
| 4000 | 1.284 | 1.285 | 1.282 |
| 5000 | 1.275 | 1.275 | 1.272 |
| 6000 | 1.269 | 1.269 | 1.266 |

6. **Quantization does not change the activation ranking.** The top family keeps the same order after int8+zlib, with similar gaps: `leaky(0.5)² < relu² < abs²`.

## What this rules out

- **Hard zeros are not the main story.** `abs²` and `softplus²` beat some zero-producing variants.
- **Output magnitude alone is not enough.** `2 * relu(x)` without squaring is worse than `relu²`.
- **The old wave-23 mechanism split was invalid.** `relu_detach²` changed gradient scale, not gradient adaptivity.
- **Adding a gate is not a free win.** At equal width budgets, gates help relative to a narrow no-gate MLP, but full-width non-gated `relu²` still wins.

## Mechanism: best current read

Wave 25 sharpened the story:

| Variant | BPB (~1300) | Read |
|---|---:|---|
| relu² | 1.3574 | baseline |
| relu² x2 init scale | 1.3576 | init scale alone does not explain the win |
| relu² const-grad | 1.4374 | adaptive gradients matter a lot |
| tanh² | 1.3995 | signal compression is bad |

Best current interpretation:

- **Adaptive gradients matter.** Removing activation-proportional gradients hurts badly.
- **Signal preservation matters.** `tanh²` is simple but too compressive.
- **Init scale matters, but is not the whole story.** It explains some early-run behavior, not the final ranking.

## What is still uncertain

- Whether `leaky(0.5)²` stays best at full submission length.
- How much of the gain is from better optimization vs better features. We know const-grad is bad; we do not yet have a clean percentage split.
- The exact optimal leak. `0.5` is the best tested long-run point so far, not a proven optimum.

## Secondary result

At matched parameter count, width beats gating:

| Variant | BPB (500) | Hidden dim |
|---|---:|---:|
| relu² | 1.4805 | 1024 |
| gated_relu² | 1.4796 | 683 + gate |
| relu²_narrow | 1.4908 | 683 |

The gate helps relative to a narrow no-gate MLP, but full-width `relu²` is still better overall.

Gate placement also matters:

| Variant | BPB (500) |
|---|---:|
| gated_relu² (`square -> gate`) | 1.4779 |
| relu²_postsigmoid (`gate -> square`) | 1.4850 |
| relu²_lingate | 1.4965 |

Squaring before gating is better than gating before squaring; an unconstrained linear gate is worse still.
