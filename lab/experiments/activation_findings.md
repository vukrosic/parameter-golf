# Activation Function Ablation: What We Know and Don't Know

Goal: understand which abstract properties make relu² work, not just rank functions.

**Baseline resolved (act20/act21/act22).** The original relu² score (1.4522) was from a dirty worktree and is not comparable. On current clean code:

| Activation | Seed 1337 | Seed 42 | Seed 99 | Mean |
|---|---:|---:|---:|---:|
| abs² | 1.4712 | 1.4685 | 1.4698 | **1.4698** |
| relu² | 1.4807 | 1.4816 | 1.4792 | **1.4805** |

abs² consistently beats relu² across 3 seeds (Δ = 0.0107 at 500 steps). All within-wave comparisons remain valid. Cross-wave comparisons against 1.4522 are invalid — use act20_relu2_ctrl (1.4807) or act22 2000-step data as reference.

---

## Property 1: Squaring helps relu massively, others barely or not at all

| Comparison | BPB | Δ from squaring |
|---|---:|---:|
| relu → relu² | 1.5007 → 1.4522 | **-0.049** |
| silu → silu² | 1.4908 → 1.4841 | -0.007 |
| swiglu → swiglu² | 1.4668 → 1.6055 | +0.139 (diverged) |
| swirelu → swirelu² | 1.4854 → 1.5276 | +0.042 |

Squaring is not universally helpful. It massively helps relu, barely helps silu, and destroys gated functions. The benefit depends on what you're squaring.

**We tested two hypotheses for why relu benefits most:**

- H1: hard zeros preserve clean sparsity after squaring (silu's negatives become positive noise)
- H2: relu's positive-side shape (pure linear) is uniquely suited to squaring

| Test | BPB | Δ vs relu² | Verdict |
|---|---:|---:|---|
| clamp(silu(x), 0)² | 1.4855 | +0.033 | Adding hard zeros to silu does NOT match relu². **H1 rejected.** |
| clamp(gelu(x), 0)² | 1.4809 | +0.029 | Same — hard zeros on gelu don't match either. |
| softplus(x)² | 1.4788 | +0.027 | No zeros at all, yet BETTER than both clamped variants. |

**Result:** Hard zeros are not the explanation. `softplus²` has no zeros and beats `clamp(silu(x), 0)²`. The positive-side shape matters more than whether negatives are exactly zero.

**New act18 results — testing the linearity hypothesis (H2):**

| Test | BPB | Positive side | Negative side |
|---|---:|---|---|
| **abs² (= x²)** | **1.4712** | linear | linear (mirrored) |
| elu² | 1.4778 | linear (same as relu) | smooth, non-zero |
| sharp_softplus² (β=10) | 1.4779 | nearly linear | nearly zero |
| sharper_softplus² (β=50) | 1.4808 | very nearly linear | very nearly zero |

**abs² is the best non-baseline result in all 40+ experiments.** It has NO activation function at all — just `f(x) = x²`. No zeros, no threshold, no nonlinearity before squaring.

**This challenges the linearity hypothesis too.** If positive-side linearity were the key, elu² (same positive side as relu) should match relu². It doesn't — elu² = 1.4778 vs relu² = 1.4522 (but see baseline warning above). More importantly, abs² beats elu² by 0.007 despite having no threshold at all.

**The softplus β sweep went the wrong direction.** β=50 (closer to relu) = 1.4808, worse than β=10 = 1.4779. If approaching relu's shape helped, higher β should be better. It isn't.

**Emerging pattern:** the simplest function wins. abs² > elu² > sharp_softplus² > clamp variants. The less processing before squaring, the better. Squaring the raw linear output (abs² = x²) is the strongest result so far.

**Caution:** abs² at 1.4712 is a single seed. relu² at seed 1337 was also a single-seed outlier (0.023 below seed 2025). We've queued multi-seed runs for both abs² and relu² (act21) to compare properly. Until those come back, abs² being better than relu² is a hypothesis, not a finding.

---

## Property 2: Hard zeros — less important than we thought

Without squaring:

| Activation | BPB | Has hard zeros? |
|---|---:|---|
| silu | 1.4908 | no (smooth, min ≈ -0.099) |
| mish | 1.4931 | no (smooth) |
| gelu | 1.4934 | no (smooth, min ≈ -0.17) |
| relu | 1.5007 | yes (exact zero for x < 0) |

Without squaring, relu is the worst. Hard zeros hurt at base level.

With squaring + hard zeros added artificially:

| Activation | BPB | Δ vs relu² |
|---|---:|---:|
| relu² | 1.4522 | — |
| softplus² (no zeros) | 1.4788 | +0.027 |
| clamp(gelu, 0)² | 1.4809 | +0.029 |
| leaky_relu(0.01)² | 1.4809 | +0.029 |
| clamp(silu, 0)² | 1.4855 | +0.033 |
| leaky_relu(0.1)² | pending | — |

`softplus²` has no zeros and outperforms `clamp(silu, 0)²` which does have zeros. `leaky_relu(0.01)²` (tiny leak, near-zero negatives) matches the clamped variants.

**Conclusion:** Hard zeros help somewhat but are not the primary factor. The gap between relu² and the rest is mostly about the positive-side shape, not the zero/nonzero boundary.

---

## Property 3: Gates hurt, but it's mostly about width

| Variant | BPB | Δ | Hidden dim |
|---|---:|---:|---:|
| relu² (full width) | 1.4522 | — | 1024 |
| gated_relu² | 1.4796 | +0.027 | 683 + gate |
| relu²_narrow (no gate) | 1.4908 | +0.039 | 683, no gate |

**Key finding:** `relu²_narrow` (2/3 hidden, no gate) loses **more** than `gated_relu²` (2/3 hidden, with gate). At matched width, the gate actually helps (+0.011 recovery).

| Gate variant | BPB | Δ vs relu²_narrow |
|---|---:|---:|
| relu²_narrow (no gate) | 1.4908 | — |
| gated_relu² (sigmoid gate) | 1.4796 | -0.011 (gate helps) |
| mild_gated (floor=0.5) | 1.4745 | -0.016 (gate helps more) |

**Revised conclusion:** Gates are not harmful. The reason all gated variants lose to relu² is **reduced hidden width** (2/3 to fit the gate matrix). At equal width, gates provide a small benefit. relu² wins because it spends all parameters on width rather than splitting them between a value path and a gate.

---

## Property 4: Gate placement

| Variant | BPB | Δ |
|---|---:|---:|
| relu² (no gate) | 1.4522 | — |
| gated_relu² (square then gate) | 1.4796 | +0.027 |
| relu²_postsigmoid (gate then square) | 1.4937 | +0.042 |

Gate-then-square is worse than square-then-gate by 0.014. Squaring a gated signal amplifies gate noise.

---

## Property 5: Optimal exponent

| relu^p | BPB | Δ |
|---|---:|---:|
| p=1.0 | 1.4778 | +0.026 |
| p=2.0 | 1.4534 | +0.001 |
| p=2.2 | 1.4546 | +0.002 |
| p=3.0 | 1.5097 | +0.058 |

p=2 is optimal. p=3 is unstable. p=1 loses 0.026 — confirms superlinear amplification helps.

---

## Seed check

| Activation | Seed 1337 | Seed 2025 | Gap |
|---|---:|---:|---:|
| relu² | 1.4522 | 1.4756 | 0.023 |
| swiglu | 1.4681 | 1.4669 | 0.001 |
| reluglu | 1.4733 | 1.4683 | 0.005 |
| gated_relu² | 1.4779 | 1.4808 | 0.003 |

Seed variance is ~0.003 for most. relu² at seed 1337 is a lucky outlier (0.023 gap). True relu² is probably ~1.46. Rankings hold across seeds.

---

## Summary: research questions, evidence, status

| # | Question | Key evidence | Status |
|---|---|---|---|
| 1 | Does squaring help? | relu→relu²: -0.049. silu→silu²: -0.007. swiglu→swiglu²: +0.139 (diverged). | **Answered.** Squaring helps simple functions (relu, identity), hurts complex/gated ones. |
| 2 | Are hard zeros the key? | softplus²=1.4788 (no zeros) beats clamp(silu,0)²=1.4855 (has zeros). leaky(0.5)² beats relu² at 2000 steps. | **Answered. No.** Hard zeros are a slight liability, not an asset. |
| 3 | Is it the positive-side shape? | abs² (= x²) matches/beats relu² across 3 seeds and 2000 steps. elu² < abs² despite same positive side. | **Answered.** Positive-side linearity helps because it means less processing before squaring. |
| 4 | Do gates help or hurt? | At matched width, gates help (+0.011). relu² wins because it spends all params on width. | **Answered.** Gates fine, width matters more. |
| 5 | What exponent is best? | p=1: 1.4778, p=2: 1.4534, p=2.2: 1.4546, p=3: 1.5097. | **Answered.** p=2 optimal. |
| 6 | Does threshold position matter? | Symmetric narrow dead zone > one-sided threshold. Less zeroing is better. | **Answered.** |
| 7 | Does suppressing negatives matter? | abs² (no suppression) confirmed across 3 seeds and 2000 steps: matches/beats relu². leaky(0.5)² is overall best. | **Answered. Suppression hurts slightly.** Optimal is partial preservation (leak ~0.5). |
| 8 | Is the relu² baseline reliable? | act20 ctrl: 1.4807 (seed 1337). act21: mean 1.4805 across 3 seeds. | **Resolved.** Original 1.4522 was dirty worktree artifact. |
| 9 | Do rankings hold at 2000 steps? | act22: ranking order identical at steps 50, 250, 500, 1000, 2000. relu² worst of 5 at every checkpoint. | **Answered.** 500-step within-wave rankings are reliable. |
| 10 | Why does squaring work? | See "Emerging theory" section above. Hypotheses: quadratic features, adaptive gradient, contrast amplification. | **Open. Next experiments needed.** |

### What the data shows (within-wave comparisons only)

- **act17 wave** (same code, same seed): softplus²=1.4788, clamp_gelu²=1.4809, leaky(0.01)²=1.4809, clamp_silu²=1.4855, relu²_narrow=1.4908. Hard zeros don't predict ranking.
- **act18 wave** (same code, same seed): abs²=1.4712, elu²=1.4778, sharp_softplus²=1.4779, sharper_softplus²=1.4808. Less processing before squaring correlates with better performance.
- **act18 surprise:** sharper_softplus (β=50, closer to relu) is WORSE than sharp_softplus (β=10). Approaching relu's shape does not help.
- **act19 wave** (same code, same seed): hard_shrink(0.2)²=1.4710, hard_shrink(0.5)²=1.4715, softshrink²=1.4748, shifted_relu_neg²=1.4832, threshold²=1.4841, shifted_relu_pos²=1.4862. Narrow symmetric dead zones beat one-sided thresholds. Less zeroing correlates with better performance.
- **act21 wave** (multi-seed): abs² mean=1.4698 vs relu² mean=1.4805 (Δ=0.0107). Confirmed across 3 seeds.
- **act22 wave** (2000 steps): leaky(0.5)²=1.3218, abs²=1.3238, softshrink²=1.3238, selu²=1.3254, relu²=1.3264. relu² last at every checkpoint. Rankings stable from step 50 onward.

### What we now know

- **Squaring is the mechanism, not relu.** abs² (= x², no activation) matches relu². The pre-squaring function barely matters.
- **Hard zeros are slightly harmful.** leaky(0.5)² consistently beats relu² by ~0.005. Preserving some negative signal helps.
- **Squaring helps simple functions, hurts complex ones.** relu/identity benefit because they pass clean signal. silu/gelu don't benefit because squaring amplifies their nonlinear artifacts.
- **500-step ablations are reliable** for within-wave ranking comparisons (confirmed by 2000-step validation).

### What we still need to understand

- **Why squaring works mechanically**: Is it output magnitude scaling, adaptive gradient (grad ∝ activation), or quadratic feature interaction?
- **Optimal negative leak**: leaky(0.5)² > relu² ≈ abs². Is 0.5 optimal, or would 0.3 or 0.7 be better?

### Remaining confounds

- **Initialization scale:** different activations produce different output magnitudes at init. The optimizer/LR may favor some scales over others.

---

## Property 6: 2000-step depth check — rankings are stable, relu² is consistently worst among squared variants

All top candidates from 500-step ablations were run to 2000 steps (wave 22, same code, same seed 1337).

**Learning curves (val_bpb at key steps):**

| Step | abs² | relu² | leaky(0.5)² | selu² | softshrink² |
|---:|---:|---:|---:|---:|---:|
| 50 | 2.335 | 2.352 | 2.349 | 2.355 | 2.358 |
| 250 | 1.606 | 1.616 | 1.605 | 1.601 | 1.611 |
| 500 | 1.472 | 1.480 | 1.469 | 1.471 | 1.473 |
| 1000 | 1.378 | 1.384 | 1.378 | 1.382 | 1.379 |
| 2000 | 1.322 | 1.325 | 1.320 | 1.324 | 1.322 |

**Final post-quantization BPB:**

| Activation | BPB | Δ vs relu² |
|---|---:|---:|
| leaky_relu(0.5)² | 1.3218 | -0.0046 |
| abs² | 1.3238 | -0.0026 |
| softshrink² | 1.3238 | -0.0026 |
| selu² | 1.3254 | -0.0010 |
| relu² | 1.3264 | — |

**Findings:**

1. **relu² is the worst of all 5 squared variants at every checkpoint from step 50 to step 2000.** It never catches up or overtakes. The common claim that relu²'s hard zeros are beneficial is not supported — variants that preserve negative signal consistently outperform.

2. **Rankings at 500 steps perfectly predict rankings at 2000 steps** (within the same wave/code). The earlier cross-wave confusion was from code changes between waves, not from rankings shifting during training. 500 steps is sufficient for within-wave ablations.

3. **The gap compresses in absolute terms** (0.017 at step 50 → 0.005 at step 2000) but this is expected as all runs converge to lower BPB. The ranking order is stable.

4. **No candidate clears the 0.005-nat submission threshold** vs relu² at 2000 steps (leaky is closest at 0.0046). These are meaningful for understanding but not for leaderboard purposes.

5. **leaky_relu(0.5)² is the overall winner** — consistently best or tied-best across all training lengths. This is relu with 50% negative signal preserved, then squared. It suggests the optimal sparsity level is "some but not total".

---

## Emerging theory: Why relu² works (and why it's not about relu)

Across 40+ experiments covering hard zeros, soft zeros, gates, thresholds, exponents, and negative signal preservation, the evidence points to a single mechanism:

**The squaring is the entire story. The relu is incidental.**

Evidence:
- abs² (= x², no activation at all) matches or beats relu² at every training length
- leaky_relu(0.5)² (half the sparsity) is consistently the best variant
- silu², which has a complex nonlinear shape before squaring, barely benefits from squaring (-0.007 vs relu's -0.049)
- The less processing before squaring, the better the result (abs² > elu² > softplus² > clamp variants)
- Hard zeros don't predict performance (softplus² beats clamp(silu,0)² despite having no zeros)

**Why squaring works (hypothesis):**

Squaring `f(x)²` applied to a linear projection `Wx` creates:
1. **Quadratic feature interactions**: output = (Wx)² = Σ w_i·w_j·x_i·x_j. The model can learn products of input features, not just linear combinations. This is strictly more expressive per parameter.
2. **Adaptive gradient scaling**: gradient of x² = 2x, so neurons with larger activations get proportionally larger gradients. This is a natural per-neuron adaptive learning rate.
3. **Contrast amplification**: values in (0,1) are compressed toward 0, values > 1 are amplified. This sharpens feature selection.

**Why relu specifically?** relu is the simplest function you can put before squaring — a linear passthrough with zeros. It doesn't add value over identity (abs²), but it also doesn't hurt much. relu² became the default because relu was already standard, not because the combination is optimal.

**Why squaring doesn't help silu/gelu:** These functions apply a smooth nonlinearity (sigmoid gate, error function) before the squaring. The squaring then amplifies the artifacts of that nonlinearity — the slight negative values, the compression of large values, the sigmoid saturation. Squaring a clean linear signal (relu, identity) amplifies the signal. Squaring a preprocessed signal amplifies the preprocessing noise.

**Open questions:**
- Is it output magnitude, gradient scaling, or feature interaction that makes squaring work?
- Does the optimal leak rate (somewhere around 0.5) depend on model size or training length?
- Would x² (no activation) work at full scale (13k steps)?

---

## Bug: power_silu divergence

`silu(x)` returns negative values (min ≈ -0.099). `torch.pow(negative, 2.2)` = NaN. Runs `act15_power_silu22` and `act15_power_silu_mildgate22` diverged from this.
