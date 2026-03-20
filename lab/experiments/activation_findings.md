# Activation Function Ablation: What We Know and Don't Know

Goal: understand which abstract properties make relu² work, not just rank functions.

**Baseline reliability warning:** relu² at seed 1337 = 1.4522, but a same-seed rerun on different code commit gave 1.4837. The original baseline ran on a dirty worktree (commit 99d69e9) — we cannot verify what code produced 1.4522. Seed 2025 gave 1.4756. **The true relu² level is uncertain until act20/act21 controls come back.** All within-wave comparisons (same commit, same wave) remain valid. Cross-wave deltas against 1.4522 are unreliable.

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
| 1 | Does squaring help? | relu→relu²: -0.049. silu→silu²: -0.007. swiglu→swiglu²: +0.139 (diverged). | Squaring helps relu massively. Not a general rule. |
| 2 | Are hard zeros the key? | softplus²=1.4788 (no zeros) beats clamp(silu,0)²=1.4855 (has zeros). | Hard zeros are not the primary factor. |
| 3 | Is it the positive-side shape? | elu² (same positive side as relu) = 1.4778. But abs² (no activation, raw x²) = 1.4712, better than elu². | Positive-side linearity helps but is not the full story. **abs² suggests less processing is better.** |
| 4 | Do gates help or hurt? | relu²_narrow=1.4908 (2/3 width, no gate) vs gated_relu²=1.4796 (2/3 width + gate). | Gates help at matched width. relu² wins on width, not because gates are bad. |
| 5 | What exponent is best? | p=1.0: 1.4778. p=2.0: 1.4534. p=2.2: 1.4546. p=3.0: 1.5097. | p=2 optimal. 4 data points. |
| 6 | Does threshold position matter? | hard_shrink(0.2)²=1.4710, hard_shrink(0.5)²=1.4715 beat shifted_relu(±0.5)²≈1.4832–1.4862. Narrow symmetric dead zone wins. | **Answered.** Symmetric > one-sided. Less zeroing is better. |
| 7 | Does suppressing negatives matter? | abs² (= x², no suppression) = 1.4712 — best non-baseline result ever. | Suppressing negatives may not help. **Pending** multi-seed confirmation (act21). |
| 8 | Is the relu² baseline reliable? | Same seed, different commits: 1.4522 vs 1.4837 (0.031 gap). Original ran on dirty worktree. | **No.** Cross-wave comparisons against 1.4522 are unreliable. act21 will establish current-code baseline. |

### What the data shows (within-wave comparisons only)

- **act17 wave** (same code, same seed): softplus²=1.4788, clamp_gelu²=1.4809, leaky(0.01)²=1.4809, clamp_silu²=1.4855, relu²_narrow=1.4908. Hard zeros don't predict ranking.
- **act18 wave** (same code, same seed): abs²=1.4712, elu²=1.4778, sharp_softplus²=1.4779, sharper_softplus²=1.4808. Less processing before squaring correlates with better performance.
- **act18 surprise:** sharper_softplus (β=50, closer to relu) is WORSE than sharp_softplus (β=10). Approaching relu's shape does not help.
- **act19 wave** (same code, same seed): hard_shrink(0.2)²=1.4710, hard_shrink(0.5)²=1.4715, softshrink²=1.4748, shifted_relu_neg²=1.4832, threshold²=1.4841, shifted_relu_pos²=1.4862. Narrow symmetric dead zones beat one-sided thresholds. Less zeroing correlates with better performance.

### What we cannot say

- Whether abs² actually beats relu² (need multi-seed comparison on same code — act21 queued).
- Why abs² works (no activation, no sparsity, no threshold — just square the linear output).
- Whether these rankings hold at 1000+ steps (act21_abs2_1000 queued).
- How much of the relu² literature explanation ("hard zeros + quadratic amplification") is applicable vs coincidental in this setup.

### Confounds we haven't controlled for

- **Initialization scale:** different activations produce different output magnitudes at init. The optimizer/LR may favor some scales over others.
- **Cross-wave code changes:** baseline 1.4522 was on a dirty worktree. Many comparisons in this file span code versions.
- **500-step horizon:** rankings may reverse at longer training, as seen with AttnRes.

---

## Bug: power_silu divergence

`silu(x)` returns negative values (min ≈ -0.099). `torch.pow(negative, 2.2)` = NaN. Runs `act15_power_silu22` and `act15_power_silu_mildgate22` diverged from this.
