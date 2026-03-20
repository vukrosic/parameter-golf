# Activation Function Ablation — Findings

**Scope:** MLP activation only (`train_gpt.py` MLP class, between up-proj and down-proj). Default: `relu²` = `proj(relu(fc(x))²)`. 40+ experiments, 500–6000 steps, seed-validated.

**Baseline (act20/act21, clean code, 3 seeds @ 500 steps; act24, 6000 steps):**

| Activation | BPB (500 steps) | BPB (6000 steps) | Post-quant BPB |
|---|---:|---:|---:|
| leaky(0.5)² | 1.4708 | **1.2659** | **1.2708** |
| relu² | 1.4805 | 1.2688 | 1.2737 |
| abs² (= x²) | 1.4698 | 1.2691 | 1.2745 |

Original relu² = 1.4522 was dirty worktree. All cross-wave comparisons use 1.4805.

---

## Finding 1: Squaring is the mechanism

Squaring massively helps simple activations, barely helps smooth ones, destroys gated ones.

| Comparison | BPB (500 steps) | Δ from squaring |
|---|---:|---:|
| relu → relu² | 1.5007 → 1.4805 | **-0.020** |
| silu → silu² | 1.4908 → 1.4841 | -0.007 |
| swiglu → swiglu² | 1.4668 → 1.6055 | +0.139 (diverged) |

Why it helps simple functions: squaring a clean linear signal (relu, identity) amplifies the signal. Squaring a preprocessed signal (silu, gelu) amplifies nonlinear artifacts. Gated functions (swiglu) already have multiplicative interaction — squaring double-gates and destabilizes.

Optimal exponent is p=2. p=1 loses 0.026, p=3 diverges.

---

## Finding 2: The pre-squaring function barely matters

At 2000 steps, all squared variants converge to within 0.005 of each other:

| Activation | BPB (2000 steps) | Δ vs relu² |
|---|---:|---:|
| leaky(0.5)² | 1.3218 | -0.0046 |
| abs² | 1.3238 | -0.0026 |
| softshrink² | 1.3238 | -0.0026 |
| selu² | 1.3254 | -0.0010 |
| relu² | 1.3264 | — |

Seed variance is ~0.003, so most of these differences are noise. The statistically meaningful finding: **squaring matters; the activation before it is second-order.**

At 500 steps, abs² leads relu² by 0.011 (confirmed across 3 seeds). **Wave 24 (6000 steps) resolved this: relu² overtakes abs² by step 5000.** The gap evolution: +0.010 at 500 → +0.005 at 1000 → +0.002 at 2000 → +0.001 at 3000 → 0.000 at 5000 → -0.0003 at 6000. abs²'s early lead was transient — likely an init-scale artifact (abs² has ~2× output variance at init, giving Adam a head start).

**leaky(0.5)² is the overall winner.** It leads at every checkpoint past step 500, and the gap vs relu² *widens* over training: 0.003 at 2000 → 0.004 at 4000 → 0.003 at 6000 (post-quant: 0.0029). It combines the init-scale advantage of abs² with mild sparsity.

---

## Finding 3: Hard zeros are not the key

Without squaring, relu (hard zeros) is the worst activation (1.5007 vs silu 1.4908). With squaring:

- softplus² (no zeros) = 1.4788 beats clamp(silu,0)² (has zeros) = 1.4855
- abs² (no zeros) = 1.4698 beats relu² (has zeros) = 1.4805
- leaky(0.5)² leads at 2000 steps

Leak rate sweep shows a clear pattern across training length:

| Leak rate | BPB (500) | BPB (2000) | BPB (6000) | Post-quant |
|---:|---:|---:|---:|---:|
| 0.0 (relu²) | 1.4806 | 1.3239 | 1.2688 | 1.2737 |
| 0.5 (leaky²) | 1.4708 | 1.3197 | 1.2659 | **1.2708** |
| 1.0 (abs²) | 1.4705 | 1.3217 | 1.2691 | 1.2745 |

At 500 steps: abs² ≈ leaky² > relu² (init-scale effect). At 6000 steps: leaky² > relu² > abs². The crossover confirms: abs²'s early advantage was init-scale, not a real feature. relu² catches up as Adam adapts. leaky(0.5)² gets the best of both — enough leak to avoid the init-scale penalty, enough zeros for mild regularization.

---

## Finding 4: Width beats gating at matched params

| Variant | BPB | Hidden dim |
|---|---:|---:|
| relu² (full width) | 1.4805 | 1024 |
| gated_relu² | 1.4796 | 683 + gate |
| relu²_narrow (no gate) | 1.4908 | 683 |

At matched width, gates help (+0.011). But relu² wins overall because it spends all parameters on width. Gate placement also matters: squaring before gating beats gating before squaring by 0.014.

---

## Finding 5: Mechanism decomposition (FLAWED — see critique)

Wave 23 tested why squaring works:

| Experiment | BPB | What it tests |
|---|---:|---|
| relu² | 1.4807 | baseline |
| relu_scaled (2·relu, no square) | 1.5022 | output magnitude → **useless** |
| relu_detach² (h·sg(h)) | 1.4963 | halved gradient scale |
| abs_detach² | 1.4848 | halved gradient scale, no zeros |

**Confirmed:** Output magnitude alone doesn't help.

**NOT confirmed:** The "50/50 adaptive gradient vs quadratic features" decomposition. The detach experiment only halves gradient magnitude — both relu² and relu_detach² still have activation-proportional gradients. The experiment tests gradient *scale*, not gradient *adaptivity*. Wave 25 (`relu2_constgrad`) fixes this with truly constant gradients.

---

## Raw data by wave

- **act17** (500 steps): softplus²=1.4788, clamp_gelu²=1.4809, leaky(0.01)²=1.4809, clamp_silu²=1.4855, relu²_narrow=1.4908
- **act18** (500 steps): abs²=1.4712, elu²=1.4778, sharp_softplus²=1.4779, sharper_softplus²=1.4808. Note: β=50 (more relu-like) is worse than β=10
- **act19** (500 steps): hard_shrink(0.2)²=1.4710, hard_shrink(0.5)²=1.4715, softshrink²=1.4748, shifted_relu_neg²=1.4832, threshold²=1.4841, shifted_relu_pos²=1.4862
- **act21** (500 steps, 3 seeds): abs² mean=1.4698, relu² mean=1.4805
- **act22** (2000 steps, learning curves): see Finding 2
- **act24** (6000 steps): relu²=1.2688 (quant 1.2737), abs²=1.2691 (quant 1.2745), leaky(0.5)²=1.2659 (quant 1.2708)

Learning curves (act22 to 2000, act24 to 6000):

| Step | abs² | relu² | leaky(0.5)² | relu²−abs² gap |
|---:|---:|---:|---:|---:|
| 50 | 2.335 | 2.352 | 2.349 | +0.017 |
| 250 | 1.606 | 1.616 | 1.605 | +0.010 |
| 500 | 1.471 | 1.481 | 1.471 | +0.010 |
| 1000 | 1.378 | 1.384 | 1.377 | +0.005 |
| 2000 | 1.322 | 1.324 | 1.320 | +0.002 |
| 3000 | 1.298 | 1.300 | 1.296 | +0.002 |
| 4000 | 1.284 | 1.285 | 1.282 | +0.001 |
| 5000 | 1.275 | 1.275 | 1.272 | 0.000 |
| 6000 | 1.269 | 1.269 | 1.266 | -0.000 |
| **Quantized** | **1.275** | **1.274** | **1.271** | **-0.001** |
