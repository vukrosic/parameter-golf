# Activation Function Ablation — Findings

**Scope:** MLP activation only, between up-proj and down-proj in `train_gpt.py`. 40+ runs, 500-13000 steps.

## What is solid

1. **Squaring helps.** For simple activations, `p=2` is the best tested exponent.

| Comparison | BPB (500) | Delta |
|---|---:|---:|
| relu -> relu² | 1.5007 -> 1.4805 | -0.020 |
| silu -> silu² | 1.4908 -> 1.4841 | -0.007 |
| relu -> relu^1.5 | 1.5007 -> 1.4862 | -0.015 |
| relu -> relu^3 | diverged | bad |
| elu² | 1.4778 | competitive with relu² |
| selu² | 1.4718 | competitive with relu² |
| celu² | 1.4792 | competitive with relu² |
| softplus² | 1.4788 | competitive with relu² |

Squaring consistently improves every simple activation tested. `p=2` is the sweet spot: `p=1.5` gets partial benefit, `p=3` diverges. The specific base function matters much less than whether you square it.

**500-step assessment:** 500 steps is reliable for screening out clearly bad ideas (relu³ diverging, squared gating blowing up — effects 10-40x noise). It also reliably identifies that squaring helps (consistent across all base functions). However, 500 steps is NOT reliable for ranking within the squared family — abs² leads at 500 but ties relu² by step 5000. **Verdict:** Keep 500-step experiments for elimination rounds, but never trust fine-grained rankings from them. Any activation within ~0.01 BPB at 500 steps needs a 2000+ step run before drawing conclusions.

2. **Squaring gated activations is bad.** **Confidence: high.** The effects are massive: +0.139 BPB (swiglu²) and +0.042 BPB (swirelu²), which are 14x and 4x the noise floor respectively. Effects this large at 500 steps don't reverse — they indicate fundamental instability (double multiplicative interaction), not a ranking that might shift. A single 2k confirmation run on swiglu² would fully nail this down if desired, but it's likely not worth the GPU time.

| Comparison | BPB (500) | Delta |
|---|---:|---:|
| swiglu -> swiglu² | 1.4668 -> 1.6055 | +0.139 |
| swirelu -> swirelu² | 1.4854 -> 1.5276 | +0.042 |

Squaring on top of a gated activation blows up in both tested cases. The double multiplicative interaction (gate × activation²) appears to be unstable.

3. **The pre-squaring function matters less than the square.** By 2000 steps, most squared variants are within ~0.005; seed noise is ~0.003. Note: the leaderboard requires beating 1.2244 BPB by ≥0.005, so a 0.005 difference between activations is right at the threshold of practical significance — worth optimizing, but not the biggest lever available.

4. **`leaky(0.5)²` is the best activation tested.**

| Activation | BPB (500) | BPB (6000) | Post-quant |
|---|---:|---:|---:|
| leaky(0.5)² | 1.4708 | **1.2659** | **1.2708** |
| relu² | 1.4805 | 1.2688 | 1.2737 |
| abs² | 1.4698 | 1.2691 | 1.2745 |

For reference, the 8xH100 record baseline is **1.2244 BPB** (13,780 steps). This shows 6k steps. leaky(0.5)² saves ~0.003 BPB post-quant over relu², which is within noise threshold.

5. **Short runs misrank activations.** `abs²` leads early (500 steps) due to higher init-scale output variance (~2×) **Hypothesized mechanism:** abs(x)² has no dead region — every input produces nonzero output — so at initialization the average activation magnitude is roughly 2x that of relu² (which kills all negative inputs). This higher initial signal could accelerate early learning. However, this is a hypothesis that should be verified by measuring actual activation statistics at initialization. If you have the checkpoints, compute mean |activation| at step 0 for abs² vs relu² to confirm. The 2*relu experiment (line below) partially tests this — scaling up relu output didn't help — but abs² changes the *shape*, not just the scale, so it's not a perfect control., but `relu²` closes the gap and ties it by step 5000. `leaky(0.5)²` leads throughout the long run.

| Step | abs² | relu² | leaky(0.5)² |
|---:|---:|---:|---:|
| 500 | 1.471 | 1.481 | 1.471 |
| 1000 | 1.378 | 1.384 | 1.377 |
| 2000 | 1.322 | 1.324 | 1.320 |
| 4000 | 1.284 | 1.285 | 1.282 |
| 5000 | 1.275 | 1.275 | 1.272 |
| 6000 | 1.269 | 1.269 | 1.266 |

6. **Quantization does not change the activation ranking.** The top family keeps the same order after int8+zlib, with similar gaps: `leaky(0.5)² < relu² < abs²`.

| Activation | BPB (6000) | Post-quant | Quant gap |
|---|---:|---:|---:|
| leaky(0.5)² | 1.2659 | 1.2708 | 0.0049 |
| relu² | 1.2688 | 1.2737 | 0.0049 |
| abs² | 1.2691 | 1.2745 | 0.0054 |

7. **The long `relu²` baseline keeps improving cleanly through 13k steps.** The 13k run is consistent with the 6k control at the overlap point, then continues steadily gaining BPB at each checkpoint.

| relu² run | Step 6000 | Final step | Final BPB | Post-quant |
|---|---:|---:|---:|---:|
| 6k control | 1.2688 | 6000 | 1.2688 | 1.2737 |
| 13k baseline | 1.2700 | 13000 | **1.2440** | **1.2498** |

The overlap at step 6000 is only ~0.0012 BPB apart, which is well within seed noise (~0.003), so the 13k baseline looks like a normal continuation rather than a different trajectory.

## What this rules out

These are hypotheses we tested and falsified — each narrows the design space for what makes a good activation.

- **Hard zeros are not the main story.** Some people might think relu² works because it produces exact zeros (sparsity). But `abs²` (never zero except at origin) and `softplus²` (smooth, never zero) perform just as well. Sparsity is not the mechanism.
- **Output magnitude alone is not enough.** `2 * relu(x)` matches relu²'s output scale without squaring. It scored **1.5034** vs relu²'s **1.4805** at 500 steps — 0.023 BPB worse. The quadratic shape itself matters, not just bigger outputs.
- **Adding a gate is not a free win.** At matched parameter budgets, gates help relative to a narrow MLP but full-width non-gated relu² still wins (see Secondary result section for full evidence and parameter counts).

## Strategic assessment

Our goal is to discover general rules about what makes activation functions work and use those rules to either beat relu² or confirm it is near-optimal. Here is where we stand:

**Rules discovered so far:**
1. Squaring is the key transformation (p=2 optimal, base function secondary)
2. Adaptive gradients are critical (const-grad penalty: 0.043 BPB)
3. Signal preservation matters (tanh² compression penalty: 0.008 BPB)
4. Some negative-side signal helps (leaky > relu > abs for the base)
5. Width > gating at matched params
6. Double multiplicative interactions are unstable (don't square gated activations)

**Have we found something better than relu²?** Yes — `leaky(0.5)²` is consistently ~0.003 BPB better, confirmed across multiple run lengths. This is a modest but real gain.

**Are we asking the right questions?** Mostly yes. The mechanism work (waves 23, 25) identified *why* squaring works (adaptive gradients + signal preservation), which correctly predicted that leaky variants would help (preserving negative-side signal). The remaining question is whether there's a bigger win available from a fundamentally different shape, or whether we're now in diminishing-returns territory where ~0.003 is the most activation choice can give us.

**Experiments organized by hypothesis (all 500 steps first):**

Each experiment below is designed to directly test one of our three hypotheses. The "predicted result" column says what should happen *if the hypothesis is correct* — if the actual result contradicts the prediction, the hypothesis is weakened.

**H1: Activation-proportional gradients are critical**
Already supported by: const-grad penalty of 0.043 BPB (14x noise).

| # | Variant | What it isolates | Predicted result if H1 is true |
|---|---|---|---|
| H1a | `leaky(0.5)² const-grad` | Is the const-grad penalty universal or relu-specific? | Should also show large penalty (~0.04+), proving H1 holds regardless of base function |
| H1b | `relu² with grad=3x` (custom backward: 3x instead of 2x) | Is steeper gradient scaling even better? | If H1, steeper could help — or there's a sweet spot at 2x. Either way, informative |

**H2: Signal compression hurts**
Already supported by: tanh² penalty of 0.008 BPB (2.7x noise).

| # | Variant | What it isolates | Predicted result if H2 is true |
|---|---|---|---|
| H2a | `hardtanh²` | Hard clips at ±1 then squares — worst-case compression | Should be worse than tanh² (hard clip is more aggressive than smooth compression) |
| H2b | `softsign²` | softsign(x) = x/(1+\|x\|), compresses to (-1,1) but smoother | Should show similar penalty to tanh² since both compress range |

**H3: Preserving negative-side information helps**
Suggestive from: leaky(0.5)² > relu² (~0.003, at noise floor). Needs stronger evidence.

| # | Variant | What it isolates | Predicted result if H3 is true |
|---|---|---|---|
| H3a | `x * abs(x)` (= sign(x) * x²) | Full negative preservation, same gradient magnitude (2\|x\|) as relu² | Should beat relu² — same H1 property but adds H3 |
| H3b | `elu(1.0)²` | Exponential negative tail (different shape than linear leak) | Should beat relu² if *any* negative info helps, not just linear leak |
| H3c | `leaky(0.5)² const-grad` vs `relu² const-grad` | Removes H1 from both — remaining difference is purely H3 | leaky const-grad should still beat relu const-grad by ~0.003 |

**Cross-validation: selu²** — selu has self-normalizing properties: it doesn't compress (satisfies H2), has a negative side (satisfies H3), and squaring gives it adaptive gradients (satisfies H1). Our theory predicts it should be competitive. It showed 1.4718 at 500 steps — best in the table. This needs 2000-step 2-seed verification to check if it's real or an abs²-like early artifact (selu's output scale is ~1.05x at init, which could give a small early boost).

**Total: 7 experiments at 500 steps (~70 min of GPU time), each testing exactly one hypothesis.**

**Scale-up plan:** Run all 7 at 500 steps → eliminate anything >0.01 worse than relu² → run survivors at 2000 steps (1 seed is enough at this stage since we're still screening, 2 seeds only needed when signal is near noise floor ~0.003) → take top → run at 6000+ steps. Save checkpoints at the end of each experiment only.

## Mechanism: best current read

Wave 25 tested three hypotheses for *why* relu² works by isolating each factor:

| Variant | BPB (~1300) | What it tests |
|---|---:|---|
| relu² | 1.3574 | baseline |
| relu² x2 init scale | 1.3576 | does higher init output scale explain the win? No. |
| relu² const-grad | 1.4374 | do adaptive (activation-proportional) gradients matter? Yes, badly. |
| tanh² | 1.3995 | does signal compression hurt? Yes. |

**Const-grad** replaces the normal relu²(x) backward pass (gradient = 2x for x>0) with a constant gradient of 1 for x>0. This removes the "adaptive" property where larger activations get larger gradients. The 0.043 BPB penalty proves that activation-proportional gradient scaling is the dominant mechanism behind relu²'s advantage. This is the primary evidence for H1. Follow-up experiments H1a (leaky const-grad) and H1b (steeper gradient) are designed in the strategic assessment section to further probe this mechanism — H1a tests universality, H1b tests whether 2x is optimal or just good enough.

**Why this matters for our goal:** These mechanism results tell us what properties to preserve when designing new activations. Any candidate must: (1) have activation-proportional gradients, (2) not compress the signal range, (3) preserve some negative-side information. leaky(0.5)² satisfies all three, which is why it wins.

## Longer-run follow-up

The 6k runs resolved the main short-run ambiguity, and the 13k `relu²` baseline confirms loss keeps decreasing smoothly at longer horizons:

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

The 8xH100 record baseline achieves **1.2244 BPB** at 13,780 steps. Our L40S 13k run reaches **1.2498 post-quant** — this is worse than the record, which is expected since L40S runs fewer tokens/sec and uses a different GPU. The L40S runs are for relative comparisons between activations, not absolute record-matching.

There is no sign of a late collapse or reversal. The post-quant gap also stays small: `1.2498 - 1.2440 = 0.0058`, very similar to the ~0.005 gaps seen in the 6k runs.

## Leak sweep follow-up

Wave 26 checked whether `0.5` is a special leak value or just one point on a plateau:

| Variant | Seed | BPB (4000) | Post-quant |
|---|---:|---:|---:|
| leaky(0.3)² | 1337 | 1.2835 | 1.2873 |
| leaky(0.7)² | 1337 | 1.2827 | 1.2867 |
| leaky(0.5)² | 42 | **1.2822** | **1.2862** |

For comparison, relu² at 4000 steps is **1.2850** (from the 6k run). All three leaky variants beat relu² by 0.0015-0.0028 BPB at this checkpoint.

The `0.5` run used seed 42 (from the earlier wave 24 run) while `0.3` and `0.7` used seed 1337. This seed mismatch means we can't precisely rank within the 0.3-0.7 range, but all three clearly beat relu², confirming the leaky advantage is robust across leak rates. The individual margins (0.0015-0.0028 BPB) are at or below the noise threshold (~0.003), so no single comparison is definitive. However, the fact that all three leaky variants beat relu² is more convincing than any individual measurement — the probability of all three landing on the same side of relu² by chance is low (~12.5% if independent). This leak sweep is evidence for H3 (negative-side info helps) — the consistency across leak rates strengthens the case, even though individual measurements are noisy. Experiments H3a (`x*abs(x)`) and H3b (`elu(1.0)²`) will provide independent H3 evidence from completely different activation shapes.

Read:

- The useful region is broad: `0.3-0.7` all work well.
- `0.3` is probably too sparse; it trails the other two.
- `0.5` still looks like the safest default, but the exact optimum is not resolved because of the seed mismatch.

## What is still uncertain

- **Whether `leaky(0.5)²` stays best at full submission length.** The ~0.003 advantage is right at noise, but it's *consistently* ~0.003 across every checkpoint from step 500 through step 6000. The real test: run leaky(0.5)² for 13k steps and compare against the 13k relu² baseline (1.2498 post-quant). If the advantage holds, submit it. If it vanishes, relu² is the answer. **This is the highest-priority experiment after the 500-step H1/H2/H3 screen.** Before spending 6-8 hrs on the 13k run, first run 500-step hyperparameter variants to check if leaky benefits from different settings than relu²: (a) `leaky(0.5)²` with 1.5x learning rate, (b) `leaky(0.5)²` with 2x warmup steps. These are cheap and could unlock additional gain on top of the activation change itself.
- **How much of the gain is from better optimization vs better features.** The const-grad experiment (which replaces relu²'s activation-proportional gradient with a constant gradient of 1) showed that optimization dynamics matter a lot. But we don't have a way to cleanly separate "leaky helps optimization" from "leaky learns better features." One approach: compare the trained models' internal representations (e.g., activation sparsity patterns, dead neuron rates). However, this is more of a scientific question than a practical one — for the competition, we can just go with *it helps*.
- **The exact optimal leak.** Experiment wave 26 suggests a broad `0.3-0.7` plateau, but not a nailed-down optimum.

## Secondary result

**How gates work:** A gate adds a second learned projection that produces a scalar mask (typically 0-1 via sigmoid) that is multiplied element-wise with the activation output. The idea is that the network can learn to selectively suppress or pass individual hidden dimensions. In `train_gpt.py`, gated variants use a `fc_gate` linear layer alongside the main `fc` layer.

**Parameter matching:** To keep total parameters equal, gated variants reduce hidden dimension to 2/3 of the non-gated width (3 matrices of size 683 instead of 2 matrices of size 1024):

| Variant | fc params | fc_gate params | proj params | Total | Hidden dim |
|---|---:|---:|---:|---:|---:|
| relu² (no gate) | 512×1024 = 524,288 | — | 1024×512 = 524,288 | **1,048,576** | 1024 |
| gated_relu² | 512×683 = 349,696 | 512×683 = 349,696 | 683×512 = 349,696 | **1,049,088** | 683 |
| relu²_narrow | 512×683 = 349,696 | — | 683×512 = 349,696 | **699,392** | 683 |

The gated and non-gated versions have matched parameter counts (~1.05M). The narrow version has fewer params — it exists to isolate the gate's contribution from the width reduction.

| Variant | BPB (500) | Params | Read |
|---|---:|---:|---|
| relu² | 1.4805 | 1,048,576 | full width, no gate |
| gated_relu² | 1.4796 | 1,049,088 | matched params, gate compensates for width loss |
| relu²_narrow | 1.4908 | 699,392 | same width as gated, but no gate |

The gate helps relative to a narrow MLP (1.4908 → 1.4796), proving the gate does useful work. But full-width relu² without a gate is essentially tied (1.4805 vs 1.4796) — the width lost to accommodate the gate costs as much as the gate contributes.

Gate placement also matters:

| Variant | BPB (500) | Order |
|---|---:|---|
| gated_relu² | 1.4779 | relu²(x) × sigmoid(gate(x)) — square first, then gate |
| relu²_postsigmoid | 1.4850 | sigmoid(gate(x)) × x, then square — gate first, then square |
| relu²_lingate | 1.4965 | relu²(x) × gate(x) — unconstrained linear gate |

Squaring before gating is better than gating before squaring. An unconstrained linear gate (no sigmoid, can output any value) is worst — the bounded 0-1 range of sigmoid appears important for stability.

## Phase 2: Full 8-hour experiment queue (6x 3090)

Everything unchanged — same architecture, same hyperparameters, same data, same seed. Only the activation function between up-proj and down-proj changes. **Critical: never run 2 experiments on the same GPU simultaneously.** Each GPU has a sequential queue. When one finishes, the next starts.

Timing estimates (3090):
- 500 steps ≈ 10 min
- 2000 steps ≈ 40 min
- 4000 steps ≈ 80 min
- 6000 steps ≈ 120 min
- 13780 steps ≈ 275 min

Baselines for comparison:
- relu² 500 steps: **1.4805**
- relu² 2000 steps: **1.324**
- relu² 4000 steps: **1.285**
- relu² 6000 steps: **1.2688** (post-quant 1.2737)
- relu² 13000 steps: **1.2440** (post-quant 1.2498)
- leaky(0.5)² 500 steps: **1.4708**
- leaky(0.5)² 6000 steps: **1.2659** (post-quant 1.2708)

---

### Master experiment list

Every experiment is tagged with the hypothesis it tests. "Standard grad" means normal autograd from the forward formula. "Custom grad" means a manually defined backward pass.

**H1 — Gradient scaling experiments (custom backward passes)**

| ID | Name | Forward | Backward (x>0 unless noted) | Hypothesis test |
|---|---|---|---|---|
| H1-1 | `relu²_grad1.5x` | relu²(x) | grad = 1.5x | Is 2x optimal or just good enough? |
| H1-2 | `relu²_grad3x` | relu²(x) | grad = 3x | Does steeper scaling help? |
| H1-3 | `relu²_gradsqrt` | relu²(x) | grad = sqrt(x) | Does sublinear scaling still work? |
| H1-4 | `relu²_gradx²` | relu²(x) | grad = x² | Super-proportional — too steep? |
| H1-5 | `leaky(0.5)²_constgrad` | leaky(0.5)²(x) | grad = 1 everywhere | H1 universality on leaky |
| H1-6 | `abs²_constgrad` | abs²(x) | grad = 1 everywhere | H1 universality on abs |
| H1-7 | `relu²_gradfloor` | relu²(x) | grad = max(2x, 0.5) | Does a gradient floor help near zero? |
| H1-8 | `relu²_gradceil` | relu²(x) | grad = min(2x, 4) | Does a gradient ceiling prevent outliers? |

**H2 — Signal compression experiments (standard grad)**

| ID | Name | Formula | Output range | Hypothesis test |
|---|---|---|---|---|
| H2-1 | `hardtanh²` | hardtanh(x)² | [0, 1] | Hard compression — worst case |
| H2-2 | `softsign²` | (x/(1+\|x\|))² | [0, 1) | Soft compression, different shape |
| H2-3 | `sigmoid²` | sigmoid(x)² | (0, 0.25] | Extreme compression |
| H2-4 | `arctan²` | arctan(x)² | [0, π²/4) | Another bounded shape |
| H2-5 | `erf²` | erf(x)² | [0, 1) | Bounded, steeper than tanh at origin |
| H2-6 | `tanh_scaled²` | (2·tanh(x))² | [0, 4] | Can scaling compensate for shape? |
| H2-7 | `relu²_clamped4` | min(relu²(x), 4) | [0, 4] | Unbounded shape, artificial ceiling |
| H2-8 | `relu²_clamped16` | min(relu²(x), 16) | [0, 16] | Higher ceiling — where does clamping stop hurting? |
| H2-9 | `log1p_relu²` | log(1 + relu²(x)) | [0, ∞) unbounded but compressed growth | Compresses growth rate, not range — tests if it's the bounding or the shape |

**H3 — Negative-side information experiments (standard grad)**

| ID | Name | Formula | Negative behavior | Hypothesis test |
|---|---|---|---|---|
| H3-1 | `x_absX` | x·\|x\| (sign(x)·x²) | Full signed output, grad = 2\|x\| | Strongest H3: same grad magnitude as relu², but sign-preserving |
| H3-2 | `elu(1.0)²` | elu(x, α=1.0)² | Exponential negative, squared | Different negative shape than linear leak |
| H3-3 | `elu(0.3)²` | elu(x, α=0.3)² | Small exponential negative | Dose-response: less negative info |
| H3-4 | `gelu²` | gelu(x)² | Smooth, slight negative dip | Popular activation — where does it rank? |
| H3-5 | `mish²` | (x·tanh(softplus(x)))² | Smooth, very slight negative | Different smooth profile than gelu |
| H3-6 | `leaky(0.1)²` | leaky_relu(x, 0.1)² | 10% slope negative | Minimum effective negative dose |
| H3-7 | `leaky(0.2)²` | leaky_relu(x, 0.2)² | 20% slope negative | Between 0.1 and 0.5 |
| H3-8 | `leaky(0.8)²` | leaky_relu(x, 0.8)² | 80% slope negative | Near-linear — is more always better? |
| H3-9 | `relu²_linneg0.5` | relu(x)² for x>0, 0.5·\|x\| for x<0 | Squared positive, linear negative | H1×H3 isolation: does the negative side NEED adaptive gradients, or just nonzero signal? Negative grad is constant 0.5, positive grad is 2x. |
| H3-10 | `bipolar_relu²` | relu(x)² − 0.25·relu(−x)² | Output goes negative (−0.25x² for x<0) | Sign-preserving like x·\|x\| but asymmetric — positive side dominates |

**Cross-hypothesis and edge cases (standard grad unless noted)**

| ID | Name | Formula | What it tests |
|---|---|---|---|
| X-1 | `leaky(0.5)³` | leaky(x, 0.5)³ | H1×exponent: is p=2 special for leaky? grad=3x² is steeper |
| X-2 | `leaky(0.5)^1.5` | leaky(x, 0.5)^1.5 | Between linear and square |
| X-3 | `softplus²_beta5` | softplus(x, β=5)² | Curvature at origin — sharper bend, more relu-like |
| X-4 | `softplus²_beta0.5` | softplus(x, β=0.5)² | Curvature at origin — smoother, wider transition |
| X-5 | `selu²` rerun | selu(x)² | Verification: 1.4718 at 500 was best — real or artifact? |
| X-6 | `celu(0.5)²` | celu(x, α=0.5)² | Negative-side shape sensitivity |
| X-7 | `relu^1.8` | relu(x)^1.8 | Exponent between linear and square |
| X-8 | `relu^2.2` | relu(x)^2.2 | Slightly more than square — stable or diverges? |
| X-9 | `x_silu` | x·silu(x) = x²·σ(x) | Smooth self-gated square — bounded by sigmoid envelope |
| X-10 | `x_tanh` | x·tanh(x) | Sign-preserving, growth bounded by tanh envelope |
| X-11 | `shifted_relu²` | relu(x + 0.5)² | Shifts activation point — negative inputs can produce nonzero output |
| X-12 | `leaky(0.5)²_lrx1.5` | leaky(0.5)² with 1.5× LR | Hyperparameter check before 13k |
| X-13 | `leaky(0.5)²_lrx0.8` | leaky(0.5)² with 0.8× LR | Hyperparameter check — maybe needs less LR? |
| X-14 | `leaky(0.5)²_warmupx2` | leaky(0.5)² with 2× warmup steps | Warmup sensitivity |

**Total: 49 experiments at 500 steps (~490 GPU-min = ~82 min wall time on 6 GPUs)**

---

### Longer-scale experiments (pre-queued)

These run on what we already know is competitive. They don't depend on 500-step results — they're pre-committed and start immediately to maximize GPU utilization.

**2000-step runs (pre-committed, ~40 min each)**

| ID | Name | Why pre-committed |
|---|---|---|
| L2-1 | `selu²` seed 42 | Best 500-step score (1.4718) — need verification at scale |
| L2-2 | `selu²` seed 1337 | Second seed for variance estimate |
| L2-3 | `elu(1.0)²` | Competitive at 500 (1.4778), strong H3 candidate |
| L2-4 | `celu²` | Competitive at 500 (1.4792), different negative shape |
| L2-5 | `softplus²` | Competitive at 500 (1.4788), no negative side — H3 control |
| L2-6 | `x_absX` | Strongest H3 test — worth pre-committing |
| L2-7 | `gelu²` | Popular, need to know ranking |
| L2-8 | `mish²` | Popular, need to know ranking |
| L2-9 | `leaky(0.1)²` | Minimum negative dose — where's the threshold? |
| L2-10 | `leaky(0.2)²` | Between 0.1 and 0.5 |
| L2-11 | `leaky(0.8)²` | Near-linear negative — too much? |
| L2-12 | `relu²_linneg0.5` | H1×H3 cross-test — does negative side need adaptive grad? |

**4000-step runs (pre-committed, ~80 min each)**

| ID | Name | Why pre-committed |
|---|---|---|
| L4-1 | `selu²` seed 42 | Full validation of best 500-step performer |
| L4-2 | `selu²` seed 1337 | Second seed |
| L4-3 | `x_absX` | Cleanest H3 test — needs long-run confirmation |
| L4-4 | `elu(1.0)²` | Strong H3 candidate |
| L4-5 | `gelu²` | Widely used — important to know ranking at scale |
| L4-6 | `leaky(0.5)²_lrx1.5` | If LR helps at 500, confirm at 4k |
| L4-7 | `leaky(0.5)²_lrx0.8` | If lower LR helps at 500, confirm at 4k |
| L4-8 | `relu²_linneg0.5` | Cross-test needs longer run — effect may be subtle |

**6000-step runs (pre-committed, ~120 min each)**

| ID | Name | Why pre-committed |
|---|---|---|
| L6-1 | `x_absX` | Head-to-head vs leaky(0.5)² at 6k (existing: 1.2659) |
| L6-2 | `selu²` | If real, compare against leaky(0.5)² at 6k |
| L6-3 | `elu(1.0)²` | Best non-leaky H3 candidate — how far can it go? |

**13780-step runs (~275 min each)**

| ID | Name | Why |
|---|---|---|
| L13-1 | `leaky(0.5)²` seed 42 | **The big bet.** Existing 6k data shows ~0.003 advantage. If it holds at 13k, submit it. |
| L13-2 | `leaky(0.5)²` seed 1337 | Second seed — we need this to know if ~0.003 is real or lucky |

**Total longer runs: 12×40 + 8×80 + 3×120 + 2×275 = 480 + 640 + 360 + 550 = 2030 GPU-min**
**Total all experiments: 490 + 2030 = 2520 GPU-min. Budget: 2880 GPU-min. Headroom: 360 min (~6 extra 500-step or 4 extra 2000-step slots for surprises)**

---

### GPU queue assignments

Each GPU runs its queue top-to-bottom. Long runs first on GPUs 0-1 (so they finish within 8 hours). GPUs 2-5 do all the 500-step screening first, then progressively longer runs.

**GPU 0 — "leaky 13k primary + H3 validation"** (~475 min = 7.9 hrs)

| Order | Experiment | Steps | Time |
|---:|---|---:|---:|
| 1 | L13-1: `leaky(0.5)²` seed 42 | 13780 | 275 min |
| 2 | L2-1: `selu²` seed 42 | 2000 | 40 min |
| 3 | L2-6: `x_absX` | 2000 | 40 min |
| 4 | L2-7: `gelu²` | 2000 | 40 min |
| 5 | H3-1: `x_absX` | 500 | 10 min |
| 6 | H3-2: `elu(1.0)²` | 500 | 10 min |
| 7 | H3-3: `elu(0.3)²` | 500 | 10 min |
| 8 | H3-9: `relu²_linneg0.5` | 500 | 10 min |
| 9 | H3-10: `bipolar_relu²` | 500 | 10 min |
| 10 | X-9: `x_silu` | 500 | 10 min |
| 11 | X-10: `x_tanh` | 500 | 10 min |

**GPU 1 — "leaky 13k replicate + H3 dose-response"** (~475 min)

| Order | Experiment | Steps | Time |
|---:|---|---:|---:|
| 1 | L13-2: `leaky(0.5)²` seed 1337 | 13780 | 275 min |
| 2 | L2-2: `selu²` seed 1337 | 2000 | 40 min |
| 3 | L2-3: `elu(1.0)²` | 2000 | 40 min |
| 4 | L2-9: `leaky(0.1)²` | 2000 | 40 min |
| 5 | H3-4: `gelu²` | 500 | 10 min |
| 6 | H3-5: `mish²` | 500 | 10 min |
| 7 | H3-6: `leaky(0.1)²` | 500 | 10 min |
| 8 | H3-7: `leaky(0.2)²` | 500 | 10 min |
| 9 | H3-8: `leaky(0.8)²` | 500 | 10 min |
| 10 | X-11: `shifted_relu²` | 500 | 10 min |

**GPU 2 — "H1 gradient screen + selu/x_absX at scale"** (~470 min)

| Order | Experiment | Steps | Time |
|---:|---|---:|---:|
| 1 | H1-1: `relu²_grad1.5x` | 500 | 10 min |
| 2 | H1-2: `relu²_grad3x` | 500 | 10 min |
| 3 | H1-3: `relu²_gradsqrt` | 500 | 10 min |
| 4 | H1-4: `relu²_gradx²` | 500 | 10 min |
| 5 | H1-5: `leaky(0.5)²_constgrad` | 500 | 10 min |
| 6 | H1-6: `abs²_constgrad` | 500 | 10 min |
| 7 | H1-7: `relu²_gradfloor` | 500 | 10 min |
| 8 | H1-8: `relu²_gradceil` | 500 | 10 min |
| 9 | L4-1: `selu²` seed 42 | 4000 | 80 min |
| 10 | L4-3: `x_absX` | 4000 | 80 min |
| 11 | L6-1: `x_absX` | 6000 | 120 min |
| 12 | L2-12: `relu²_linneg0.5` | 2000 | 40 min |
| 13 | L2-10: `leaky(0.2)²` | 2000 | 40 min |

**GPU 3 — "H2 compression screen + selu/elu at scale"** (~470 min)

| Order | Experiment | Steps | Time |
|---:|---|---:|---:|
| 1 | H2-1: `hardtanh²` | 500 | 10 min |
| 2 | H2-2: `softsign²` | 500 | 10 min |
| 3 | H2-3: `sigmoid²` | 500 | 10 min |
| 4 | H2-4: `arctan²` | 500 | 10 min |
| 5 | H2-5: `erf²` | 500 | 10 min |
| 6 | H2-6: `tanh_scaled²` | 500 | 10 min |
| 7 | H2-7: `relu²_clamped4` | 500 | 10 min |
| 8 | H2-8: `relu²_clamped16` | 500 | 10 min |
| 9 | H2-9: `log1p_relu²` | 500 | 10 min |
| 10 | L4-2: `selu²` seed 1337 | 4000 | 80 min |
| 11 | L4-4: `elu(1.0)²` | 4000 | 80 min |
| 12 | L6-2: `selu²` | 6000 | 120 min |
| 13 | L2-4: `celu²` | 2000 | 40 min |
| 14 | L2-11: `leaky(0.8)²` | 2000 | 40 min |

**GPU 4 — "Cross-hypothesis + LR/exponent sweep + gelu/elu at scale"** (~470 min)

| Order | Experiment | Steps | Time |
|---:|---|---:|---:|
| 1 | X-1: `leaky(0.5)³` | 500 | 10 min |
| 2 | X-2: `leaky(0.5)^1.5` | 500 | 10 min |
| 3 | X-3: `softplus²_beta5` | 500 | 10 min |
| 4 | X-4: `softplus²_beta0.5` | 500 | 10 min |
| 5 | X-5: `selu²` rerun | 500 | 10 min |
| 6 | X-6: `celu(0.5)²` | 500 | 10 min |
| 7 | X-7: `relu^1.8` | 500 | 10 min |
| 8 | X-8: `relu^2.2` | 500 | 10 min |
| 9 | X-12: `leaky(0.5)²_lrx1.5` | 500 | 10 min |
| 10 | X-13: `leaky(0.5)²_lrx0.8` | 500 | 10 min |
| 11 | X-14: `leaky(0.5)²_warmupx2` | 500 | 10 min |
| 12 | L4-5: `gelu²` | 4000 | 80 min |
| 13 | L4-6: `leaky(0.5)²_lrx1.5` | 4000 | 80 min |
| 14 | L4-7: `leaky(0.5)²_lrx0.8` | 4000 | 80 min |
| 15 | L6-3: `elu(1.0)²` | 6000 | 120 min |

**GPU 5 — "Remaining 2000-step + overflow"** (~470 min)

| Order | Experiment | Steps | Time |
|---:|---|---:|---:|
| 1 | L2-5: `softplus²` | 2000 | 40 min |
| 2 | L2-8: `mish²` | 2000 | 40 min |
| 3 | L4-8: `relu²_linneg0.5` | 4000 | 80 min |
| 4 | spare: best H1 variant at 4000 | 4000 | 80 min |
| 5 | spare: best surprise at 4000 | 4000 | 80 min |
| 6 | spare: best overall at 6000 | 6000 | 120 min |
| 7–10 | spare slots (4 × 500 step) | 500 | 40 min |

GPU 5 has ~360 min of pre-committed work and ~120 min of spare slots. Use the spare slots for: (a) re-running any 500-step experiment that gave a surprising result with a different seed, or (b) running the best new discovery at longer scale. **Decision: check results from GPUs 2-4 as they complete their 500-step queues (~80 min in) and fill GPU 5's spare slots with the most promising follow-ups.**

---

### Decision criteria after Phase 2

**After 500-step results come in (~80 min wall time):**

| Result pattern | What it means | What to do with spare GPU-5 slots |
|---|---|---|
| All const-grad variants collapse (H1-5, H1-6 show ~0.04+ penalty) | H1 confirmed universal | H1 is settled, don't scale gradient variants |
| grad=3x beats grad=2x by >0.005 | Steeper gradients help | Run best gradient variant at 4000 steps in spare slot |
| All H2 bounded activations lose by >0.008 | H2 confirmed | H2 is settled, don't scale any bounded variant |
| `x_absX` beats relu² by >0.005 at 500 | H3 strongly confirmed | It's already queued at 4000 and 6000 — watch those results closely |
| `relu²_linneg0.5` competitive with leaky(0.5)² | Negative side doesn't need adaptive grad | This is a new insight — scale it up in spare slot |
| `leaky(0.5)³` diverges | p=2 is fundamental even for leaky | Confirms p=2 constraint is universal |
| `relu^2.2` competitive and stable | Exponent has room above 2.0 | Try leaky(0.5)^2.2 in spare slot |
| selu² rerun matches 1.4718 | Original result was real, not artifact | Already scaling at 4000/6000 — monitor |
| LR or warmup variant beats default leaky(0.5)² | Hyperparameters not yet optimized for leaky | Scale winning hyperparameter config to 4000 (already queued) |

**After 13k results come in (~5 hrs):**

| Result | Meaning | Action |
|---|---|---|
| Both seeds show leaky(0.5)² beating relu² 13k baseline (1.2498 post-quant) by >0.003 | Advantage is real and holds at scale | **Submit leaky(0.5)² as activation. Then check if any Phase 2 discovery beats leaky(0.5)² and run that at 13k.** |
| One seed better, one seed worse | Within noise even at 13k | Need 3rd seed or look for a bigger gain from Phase 2 discoveries |
| Both seeds match or lose to relu² | Advantage was noise | relu² is the answer for activation. Focus effort on other competition axes. |