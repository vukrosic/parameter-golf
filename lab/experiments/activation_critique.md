# Critique of Activation Findings

## Flaw 1 (HIGH): Mechanism decomposition is broken

**Claim:** "~50% adaptive gradient, ~50% quadratic features."

**Problem:** Both relu² and relu_detach² have activation-proportional gradients:

| Variant | Gradient w.r.t. x |
|---|---|
| relu² | 2·relu(x)·1(x>0) — **adaptive** |
| relu_detach² (h·sg(h)) | relu(x)·1(x>0) — **still adaptive** |
| plain relu | 1(x>0) — constant |

The detach experiment only halves gradient *scale*, not *adaptivity*. Both variants give larger gradients to larger activations. The experiment tests "what if effective LR is halved" not "what if we remove adaptive gradients."

**Fix:** Wave 25 `relu2_constgrad` uses a custom autograd function: forward = relu(x)², backward = 1(x>0) (truly constant, not proportional to activation).

## Flaw 2 (HIGH): Init scale confound may explain convergence trends

relu² zeros ~50% of activations → output variance ≈ half of abs² at init. The "relu² catches up over training" trend could be Adam calibrating to a smaller scale, not regularization from sparsity. Same confound infects the leak rate sweep (more leak = larger variance = faster early convergence).

**Fix:** Wave 25 `relu2_initscale` multiplies relu² output by 2 to match abs² variance. If early gap disappears, init scale was the confound.

## Flaw 3 (MEDIUM): "Simpler = better" is post-hoc

abs² > elu² > sharp_softplus² could equally be "more signal preservation = better" — different framing, different predictions.

**Fix:** Wave 25 `tanh2`. Tanh is simple but compresses signal. Bad result → signal preservation wins.

## Flaw 4 (MEDIUM): Rankings between squared variants are within noise

Seed variance ~0.003. Most squared-variant differences at 2000 steps (Δ < 0.005) are indistinguishable. The clean finding is "squaring matters; pre-squaring function doesn't" — not detailed rankings.

## Flaw 5 (LOW-MEDIUM): "Quadratic feature interaction" overstates

(Wx)² is a rank-1 quadratic form, not full quadratic. SwiGLU also has multiplicative interaction (σ(Wx)⊙Vx) but doesn't benefit from squaring — contradicts the "products of features" explanation.

## Flaw 6 (LOW): Convergence extrapolation unjustified

Gap: 0.028→0.010→0.008→0.006→0.003. Equally consistent with decay toward nonzero asymptote (~0.002) as toward zero. Wave 24 (6000 steps, running) resolves this directly.

## Flaw 7 (MEDIUM): Effective capacity not matched

relu² uses ~50% of neurons, abs² uses ~100%. We're comparing activations at different effective widths. This doesn't invalidate findings but adds context — the convergence trend could be "optimizer compensating for wasted capacity."
