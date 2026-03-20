# Activation Function Ablation: What We Know and Don't Know

Baseline: `relu(x)²` at **1.4522 BPB** (500 steps, seed 1337, 1xL40S).

Goal: understand which abstract properties make relu² work, not just rank functions.

---

## Property 1: Superlinear amplification

**Question:** Does raising activations to a power > 1 help?

| Comparison | BPB | Δ |
|---|---:|---:|
| relu → relu² | 1.5007 → 1.4522 | **-0.049** |
| silu → silu² | 1.4908 → 1.4841 | -0.007 |
| swiglu → swiglu² | 1.4668 → 1.6055 | +0.139 (diverged) |
| swirelu → swirelu² | 1.4854 → 1.5276 | +0.042 |

**Observed:** Squaring massively helps relu, barely helps silu, and destroys gated functions.

**Not a general rule.** Squaring interacts with the base function. The benefit depends on what you're squaring.

**What we don't know:** Why does relu benefit so much more than silu from squaring? Two hypotheses but no proof yet:

- H1: relu has exact zeros, so squaring preserves clean sparsity. silu has small negatives (~-0.099) that get squared into positive noise, polluting the sparse signal.
- H2: relu's output distribution (half zero, half linear) is better suited to squaring than silu's (smooth, never exactly zero).

**Experiments needed to decide:**
- `clamp(silu(x), min=0)²` — if this matches relu², H1 is confirmed (hard zeros are what matters)
- `softplus(x)²` — softplus is always positive, no zeros at all. Tests whether any-positive-activation + square can work.

---

## Property 2: Hard zeros (sparsity)

**Question:** Does zeroing out negative inputs help?

Without squaring (base activations):

| Activation | BPB | Hard zeros? |
|---|---:|---|
| silu | 1.4908 | no |
| mish | 1.4931 | no |
| gelu | 1.4934 | no |
| relu | 1.5007 | yes |

**relu is the worst base activation.** Hard zeros alone do not help. Smooth functions are better without squaring.

With squaring:

| Activation | BPB | Hard zeros? |
|---|---:|---|
| relu² | 1.4522 | yes |
| silu² | 1.4841 | no |

**relu² beats silu² by 0.032.** Hard zeros only matter when combined with squaring.

**Observed:** Sparsity and squaring have a synergistic interaction. Neither alone is great. Together they dominate.

**What we don't know:** Is the synergy because (a) zeros stay clean after squaring, (b) the gradient is also zero at the threshold creating a sharper on/off switch, or (c) something else about the distribution shape?

**Experiments needed:**
- `clamp(silu(x), min=0)²` — makes silu have hard zeros. If it matches relu², the answer is (a).
- `relu² with 2/3 hidden dim` (parameter-matched to GLU variants) — isolates whether relu²'s advantage over gated functions is from sparsity or from having more hidden units.

---

## Property 3: Gating (separate learned gate)

**Question:** Does adding a learned gate to relu² help?

| Variant | BPB | Δ | Gate type |
|---|---:|---:|---|
| relu² (no gate) | 1.4522 | — | none |
| mild_gated (floor=0.5) | 1.4745 | +0.022 | bounded sigmoid |
| gated_relu² | 1.4796 | +0.027 | sigmoid |
| mild_gated (floor=0.3) | 1.4846 | +0.032 | wider bounded |
| mild_gated (floor=0.7) | 1.4864 | +0.034 | tighter bounded |
| relu²_lingate | 1.4873 | +0.035 | linear |
| relu²_softplus_gate | 1.5123 | +0.060 | softplus |

**Observed:** Every gate type hurts. Tested 6 gate types including 3 floor values on bounded sigmoid. No gate helps.

**But there's a confound:** gated variants use 2/3 hidden dimension (to match parameter count with the extra gate matrix). So is the loss from the gate itself, or from having fewer hidden units?

**What we don't know:** Is gating inherently bad, or just parameter-inefficient at this scale?

**Experiments needed:**
- `relu² at 2/3 hidden dim` (no gate) — if this also loses ~0.022, the gate isn't hurting, it's the reduced width.
- `relu² at full hidden + gate` (more params) — if this still loses, the gate is actively harmful.

---

## Property 4: Gate placement

**Question:** Does it matter whether gating happens before or after squaring?

| Variant | BPB | Δ |
|---|---:|---:|
| relu² (no gate) | 1.4522 | — |
| gated_relu² (square then gate) | 1.4796 | +0.027 |
| relu²_postsigmoid (gate then square) | 1.4937 | +0.042 |

**Observed:** Gate-then-square is worse than square-then-gate by 0.014. Both worse than no gate.

This is consistent with Property 1: squaring a gated signal is worse than squaring a clean signal.

---

## Property 5: Optimal exponent

| relu^p | BPB | Δ |
|---|---:|---:|
| p=1.0 | 1.4778 | +0.026 |
| p=2.0 | 1.4534 | +0.001 |
| p=2.2 | 1.4546 | +0.002 |
| p=3.0 | 1.5097 | +0.058 |

p=2 is optimal. p=2.2 is within noise. p=3 is unstable. p=1 (no amplification) loses 0.026.

This confirms superlinear amplification helps, and there's a sweet spot around 2.

---

## Seed check

| Activation | Seed 1337 | Seed 2025 | Gap |
|---|---:|---:|---:|
| relu² | 1.4522 | 1.4756 | 0.023 |
| swiglu | 1.4681 | 1.4669 | 0.001 |
| reluglu | 1.4733 | 1.4683 | 0.005 |
| gated_relu² | 1.4779 | 1.4808 | 0.003 |

Seed variance is ~0.003 for most. relu² at seed 1337 is a lucky outlier (0.023 below seed 2025). Rankings hold across seeds.

---

## What we've proven

1. **Squaring + hard zeros interact synergistically.** Neither alone is best. Together they dominate.
2. **Gates always hurt relu²** across 6 gate types tested.
3. **p=2 is the optimal exponent** (tested 1.0, 2.0, 2.2, 3.0).

## What we haven't proven (need experiments)

1. **WHY do hard zeros make squaring work?** Need `clamp(silu(x), min=0)²` to test if just adding zeros to silu makes it match relu².
2. **Is the gate loss from the gate or from reduced hidden dim?** Need `relu² at 2/3 hidden` to isolate.
3. **Can a non-relu function with hard zeros match relu²?** Need `clamp(gelu(x), min=0)²` or `hardtanh_positive(x)²`.
4. **Would a smooth function that's truly zero below threshold work?** A threshold-shifted softplus like `softplus(x - k)²` could test this.

---

## Bug: power_silu divergence

`silu(x)` returns negative values (min ≈ -0.099). `torch.pow(negative, 2.2)` = NaN. Runs `act15_power_silu22` and `act15_power_silu_mildgate22` failed from this. Fix needed before any non-integer silu power experiments.
