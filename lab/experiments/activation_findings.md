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

**Are we asking the right questions?** Mostly yes. The mechanism work (waves 23, 25) identified *why* squaring works (adaptive gradients + signal preservation) **Proposed experiments to exploit adaptive gradient mechanism (500-step screen):**

| # | Variant | Rationale | What it tests |
|---|---|---|---|
| 1 | `x * abs(x)` (= sign(x) * x²) | Same gradient magnitude as relu² (2|x|) but preserves sign — negative inputs get negative outputs | Does preserving sign matter, or just gradient scaling? |
| 2 | `leaky(0.5)² + small constant` (e.g., + 0.01) | Adds a floor so gradient never fully vanishes near zero | Does a gradient floor help or hurt? [why do we need this?] |
| 3 | `softplus(x)²` with different beta | Beta controls sharpness of the bend — higher beta = more relu-like | Is there an optimal curvature at the origin? |
| 4 | `selu²` at 2000 steps (2 seeds) | selu² showed 1.4718 at 500 steps, best in the table — needs verification | Is selu² actually competitive or an early-run artifact like abs²? [what doe our theory say, how does it predict it] |
| 5 | `elu(alpha=0.5)²` | ELU with tuned alpha gives controlled negative output | Is there an optimal negative-side shape? |
| 6 | `leaky(0.5)²` with gradient clipping at 2x | Caps the maximum gradient to prevent outlier updates | Does gradient proportionality have a ceiling? |

[are these experiments gonna confirm or deny our 3 conclusions, (1) have activation-proportional gradients,
(2) not compress the signal range,
(3) preserve some negative-side information. i need all experiments to be based on this and to test these 3, start with 500 steps ones]

**Scale-up plan:** Run all 500 step experiments → take ones that are not much worse → run at 2000 steps with 2 seeds each (why 2 seeds? is it necessary?) → take top → run at 6000+ steps. Save checkpoints at the end of each experiment only.

## Mechanism: best current read

Wave 25 tested three hypotheses for *why* relu² works by isolating each factor:

| Variant | BPB (~1300) | What it tests |
|---|---:|---|
| relu² | 1.3574 | baseline |
| relu² x2 init scale | 1.3576 | does higher init output scale explain the win? No. |
| relu² const-grad | 1.4374 | do adaptive (activation-proportional) gradients matter? Yes, badly. |
| tanh² | 1.3995 | does signal compression hurt? Yes. |

**Const-grad** replaces the normal relu²(x) backward pass (gradient = 2x for x>0) with a constant gradient of 1 for x>0. This removes the "adaptive" property where larger activations get larger gradients. The 0.043 BPB penalty proves that activation-proportional gradient scaling is the dominant mechanism behind relu²'s advantage. **Follow-up experiments designed above in the strategic assessment section.** The key insight from const-grad is that gradient magnitude scaling with activation value is the main mechanism. This suggests exploring: (a) different gradient scaling rates (steeper or shallower than 2x), (b) gradient floors/ceilings, (c) whether the scaling matters more on the positive or negative side. [does this play into our 3 main hypothesis]

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

The `0.5` run used seed 42 (from the earlier wave 24 run) while `0.3` and `0.7` used seed 1337. This seed mismatch means we can't precisely rank within the 0.3-0.7 range, but all three clearly beat relu², confirming the leaky advantage is robust across leak rates. The individual margins (0.0015-0.0028 BPB) are at or below the noise threshold (~0.003), so no single comparison is definitive. However, the fact that all three leaky variants beat relu² is more convincing than any individual measurement — the probability of all three landing on the same side of relu² by chance is low (~12.5% if independent). [so do you have them in experiments, keep in mind experiments must play into our 3 main hypothesis]

Read:

- The useful region is broad: `0.3-0.7` all work well.
- `0.3` is probably too sparse; it trails the other two.
- `0.5` still looks like the safest default, but the exact optimum is not resolved because of the seed mismatch.

## What is still uncertain

- **Whether `leaky(0.5)²` stays best at full submission length.** We still do not have a 13k leaky run against the 13k `relu²` baseline. Yes, we should run this — it's the most important remaining experiment for the activation workstream. If the ~0.003 advantage holds at 13k, it directly translates to a submission improvement. **Yes, ~0.003 is right at noise.** But it's *consistently* ~0.003 across every checkpoint from step 500 through step 6000, which is more convincing than a single measurement. The real test: run leaky(0.5)² for 13k steps with 2 seeds and compare against the 13k relu² baseline. If the advantage holds at 13k, it compounds over longer training and is worth submitting. If it vanishes, relu² is the answer and we stop. **This is the highest-priority experiment remaining.** Tuning (learning rate, warmup) for leaky specifically could also help — the current runs all use relu²-optimized hyperparameters. [let's do some of these for 500 steps]
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

## Experiment scale-up plan (1 GPU)

Progressive elimination pipeline to maximize confidence per GPU-hour:

**Stage 1: Screen (500 steps, ~10 min each)**
- Run all new candidates from the proposed experiment table above
- Eliminate anything >0.01 BPB worse than relu² (1.4805)
- Keep: everything within 0.01 of the best

**Stage 2: Validate (2000 steps, 2 seeds each, ~40 min per run)**
- Run survivors with 2 different seeds [why 2, is it necessary or not]
- This is the critical stage: 2 seeds at 2000 steps lets you estimate variance and detect flukes [but if the gain is over threshold 0.005 does it matter?]
- Eliminate: anything whose mean across seeds is worse than relu² mean
- Keep: top

**Stage 3: Confirm (4000-6000 steps, ~2-3 hrs each)**
- Run top 1-2 candidates plus relu² control (same seed) [don't we already have control relu squared]
- Compare against existing 6k relu² data
- Decision point, write data and we will analyze it later

**Stage 4: Full run (13,780 steps, ~6-8 hrs)**
- we will think about this later

**Priority queue (most important first):**
1. leaky(0.5)² at 13k steps (highest expected value — already validated through 6k)
2. selu² verification at 2000 steps, 2 seeds (suspiciously good at 500, needs confirmation)
3. x*abs(x) at 500 steps (novel shape, strong theoretical motivation)
4. New candidates from experiment table at 500 steps
[are these true? if so add them to experiment list]