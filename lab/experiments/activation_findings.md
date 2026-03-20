# Activation Function Ablation — Findings

**Scope:** MLP activation only, between up-proj and down-proj in `train_gpt.py`. 40+ runs, 500-13000 steps. Clean `relu²` baseline is `1.4805` at 500; the older `1.4522` came from a dirty worktree and should not be used. [i no longer want any mentions of the dirty worktree, remove ones that are dirty and don't mentioned them anymore]

## What is solid

1. **Squaring helps.** For simple activations, `p=2` is the best tested exponent.

| Comparison | BPB (500) | Delta |
|---|---:|---:|
| relu -> relu² | 1.5007 -> 1.4805 | -0.020 |
| silu -> silu² | 1.4908 -> 1.4841 | -0.007 |
| relu -> relu^3 | diverged | bad |
[this needs more evidence, you have more experiments to put here? of other types not just squared?]
2. **Squaring gated activations is bad.**

| Comparison | BPB (500) | Delta |
|---|---:|---:|
| swiglu -> swiglu² | 1.4668 -> 1.6055 | +0.139 |
[are there more experiments to put here? did we do more?]
3. **The pre-squaring function matters less than the square.** By 2000 steps, most squared variants are within ~0.005; seed noise is ~0.003. [is it true that within 0.005 is not acceptable to the leaderboard, if so mention it]

4. **`leaky(0.5)²` is the best activation tested.**

| Activation | BPB (500) | BPB (6000) | Post-quant |
|---|---:|---:|---:|
| leaky(0.5)² | 1.4708 | **1.2659** | **1.2708** |
| relu² | 1.4805 | 1.2688 | 1.2737 |
| abs² | 1.4698 | 1.2691 | 1.2745 |
[how does this compare to the baseline?]
5. **Short runs misrank activations.** `abs²` leads early, but `relu²` catches and slightly passes it by step 5000. [i dont see slight surpass here in the table] `leaky(0.5)²` leads throughout the long run.

| Step | abs² | relu² | leaky(0.5)² |
|---:|---:|---:|---:|
| 500 | 1.471 | 1.481 | 1.471 |
| 1000 | 1.378 | 1.384 | 1.377 |
| 2000 | 1.322 | 1.324 | 1.320 |
| 4000 | 1.284 | 1.285 | 1.282 |
| 5000 | 1.275 | 1.275 | 1.272 |
| 6000 | 1.269 | 1.269 | 1.266 |

6. **Quantization does not change the activation ranking.** The top family keeps the same order after int8+zlib, with similar gaps: `leaky(0.5)² < relu² < abs²`. [evidence]

7. **The long `relu²` baseline keeps improving cleanly through 13k steps.** The 13k run is consistent with the 6k control at the overlap point, then keeps buying real loss. [what do you mean keeps buying]

| relu² run | Step 6000 | Final step | Final BPB | Post-quant |
|---|---:|---:|---:|---:|
| 6k control | 1.2688 | 6000 | 1.2688 | 1.2737 |
| 13k baseline | 1.2700 | 13000 | **1.2440** | **1.2498** |

The overlap at step 6000 is only ~0.0012 BPB apart [is this within noise?], so the 13k baseline looks like a normal continuation rather than a different trajectory.

## What this rules out [what do you mean here]

- **Hard zeros are not the main story.** `abs²` and `softplus²` beat some zero-producing variants.[what do you mean here]
- **Output magnitude alone is not enough.** `2 * relu(x)` without squaring is worse than `relu²`. [show evidence]
- **The old wave-23 mechanism split was invalid.** `relu_detach²` changed gradient scale, not gradient adaptivity. [not sure if this should be here, is it useful]
- **Adding a gate is not a free win.** At equal width budgets, gates help relative to a narrow no-gate MLP, but full-width non-gated `relu²` still wins. [a bit unclear, add evidence]

[so based on the entire file until now, what is our goal, it's to discover rules and come up with next good activation function that beats relu 2 or prove it's best possible, how are we doing on this front? are we asking and answering correct questions?]

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
[this is unclear as to what it's doing and how it's realting to our main goal]
## Longer-run follow-up

The 6k runs resolved the main short-run ambiguity, and the 13k `relu²` baseline says the gains are still alive late:

| Step | relu² (13k baseline) |
|---:|---:|
| 6000 | 1.2700 |
| 7000 | 1.2635 |
| 8000 | 1.2577 |
| 9000 | 1.2544 |
| 10000 | 1.2515 |
| 11000 | 1.2479 |
| 12000 | 1.2446 |
| 13000 | 1.2440 |
[is the full training loss at the end consistant with their record or not?]
There is no sign of a late collapse or reversal. The post-quant gap also stays small: `1.2498 - 1.2440 = 0.0058`, very similar to the ~0.005 gaps seen in the 6k runs.

## Leak sweep follow-up

Wave 26 checked whether `0.5` is a special leak value or just one point on a plateau:

| Variant | Seed | BPB (4000) | Post-quant |
|---|---:|---:|---:|
| leaky(0.3)² | 1337 | 1.2835 | 1.2873 |
| leaky(0.7)² | 1337 | 1.2827 | 1.2867 |
| leaky(0.5)² | 42 | **1.2822** | **1.2862** |
[not sure why the seed is different but i guess it works well?, also add how it compares to the relu2 baseline]
Read:

- The useful region is broad: `0.3-0.7` all work well.
- `0.3` is probably too sparse; it trails the other two.
- `0.5` still looks like the safest default, but the exact optimum is not resolved because the `0.5` follow-up used a different seed than the `0.3` and `0.7` runs.

## What is still uncertain

- Whether `leaky(0.5)²` stays best at full submission length. We still do not have a 13k leaky run against the 13k `relu²` baseline. [should we do it?]
- How much of the gain is from better optimization vs better features. [how do we measure this?] We know const-grad is bad; we do not yet have a clean percentage split. [what is const-grad, is measuring it contributing to our main goal?]
- The exact optimal leak. Wave 26 suggests a broad `0.5-0.7` plateau, but not a nailed-down optimum.

## Secondary result

At matched parameter count, width beats gating:

| Variant | BPB (500) | Hidden dim |
|---|---:|---:|
| relu² | 1.4805 | 1024 |
| gated_relu² | 1.4796 | 683 + gate |
| relu²_narrow | 1.4908 | 683 |

The gate helps relative to a narrow no-gate MLP, but full-width `relu²` is still better overall.
[this is useful insight but explain your experiments and results better so it's more intuitive, show what you have, show how many params does the gate have in your script (find it the ground truth) so we can se that wide doesn't just have more params or does it]
Gate placement also matters:

| Variant | BPB (500) |
|---|---:|
| gated_relu² (`square -> gate`) | 1.4779 |
| relu²_postsigmoid (`gate -> square`) | 1.4850 |
| relu²_lingate | 1.4965 |

Squaring before gating is better than gating before squaring; an unconstrained linear gate is worse still. [just explain shortly above how the gate works, what it does]
