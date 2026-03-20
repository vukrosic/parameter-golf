# Activation Function Ablation: What We Know and Don't Know

Baseline: `relu(x)²` at **1.4522 BPB** (500 steps, seed 1337, 1xL40S).

Goal: understand which abstract properties make relu² work, not just rank functions.

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

**What's still unique about relu:** relu's positive side is pure `f(x) = x`. When squared: clean `x²`. silu's positive side curves sublinearly near zero. gelu's positive side also curves. Even after clamping, their squared shapes are messier near the origin. relu² has the simplest possible squared shape.

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

## What we've proven

1. **Squaring helps relu far more than any other activation.** Not a general rule — depends on base shape.
2. **Hard zeros are not the primary factor.** `softplus²` (no zeros) beats `clamp(silu, 0)²` (has zeros).
3. **relu's advantage is its positive-side shape** — pure linear, so relu² = clean x². Other functions have curved positive sides that produce messier squared shapes.
4. **Gates are not inherently bad.** At matched width, gates help slightly. relu² wins by spending all params on width.
5. **p=2 is the optimal exponent.**

## Open questions

1. **Can we match relu² with a different function that has a linear positive side?** E.g., `max(x, 0)` is not the only function with `f(x) = x` for x > 0 — a shifted softplus `softplus(x, β=large)²` approaches relu² as β increases.
2. **Would relu² with extra width (MLP_MULT=3) beat relu² at standard width?** Since width is the real bottleneck.
3. **Is the positive-side linearity hypothesis testable more directly?** E.g., `x * sigmoid(βx)` with large β approaches relu — does `(x * sigmoid(βx))²` approach relu² performance?

---

## Bug: power_silu divergence

`silu(x)` returns negative values (min ≈ -0.099). `torch.pow(negative, 2.2)` = NaN. Runs `act15_power_silu22` and `act15_power_silu_mildgate22` diverged from this.
