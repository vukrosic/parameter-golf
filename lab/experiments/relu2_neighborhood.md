# ReLU2 Neighborhood Sweep

## Why this lane

`relu2` is already a strong default in this repo. The point of this sweep is not to
replace it with random activations, but to test which part of it matters:

- hard sparsity from ReLU
- quadratic amplification on surviving activations
- no gate path

SwiGLU, by contrast, brings:

- a learned multiplicative gate
- smooth gradients from `silu`
- softer selection than ReLU

So the most relevant neighborhood is not generic activation search. It is:

- `relu2`
- `relu2` plus a mild learned gate
- `relu2` plus a smooth positive gate
- ReLU value path plus SwiGLU gate
- `relu2` core plus SwiGLU gate
- `swiglu2` as the more gate-heavy reference

## Run set

- `act4_relu2_ctrl_500`: control
- `act4_gated_relu2_500`: existing sigmoid-gated `relu2`
- `act4_swirelu_500`: ReLU value path with `silu` gate
- `act4_swirelu2_500`: ReLU² core with `silu` gate
- `act4_relu2_softplus_gate_500`: ReLU² with positive smooth gate
- `act4_swiglu2_500`: gate-heavy squared SwiGLU reference

## Decision rule

At `500` steps:

- keep variants that beat `relu2`
- kill anything clearly behind by `>= 0.01` BPB
- if one gated ReLU² variant wins, extend it to `1000+` before inventing more variants

## Phase 2: Intersection Sweep

After the first wave, run a more explicit factorization of the design space:

- value path: `relu`, `relu2`, `silu`
- gate type: linear, sigmoid, `silu`, softplus
- square placement: before gate vs after gate

Second-wave runs:

- `act5_swiglu_ctrl_500`: standard SwiGLU reference
- `act5_silu2_500`: smooth single-path squared reference
- `act5_reluglu_500`: ReLU value path + linear gate
- `act5_relu2_lingate_500`: ReLU² + linear gate
- `act5_relu2_postsigmoid_500`: gate first, square second

This wave is meant to answer:

- is the gain coming from the square itself?
- is it the gate, not the square?
- does ReLU sparsity matter more than SwiGLU smoothness?
- is `relu2` best because it squares after hard selection?

## Current Results Snapshot

`500`-step results seen so far:

| Run | Activation | BPB | Read |
|---|---|---:|---|
| `baseline_500` | baseline training config | `1.4522` | still strongest known `500`-step reference |
| `act4_gated_relu2_500` | `gated_relu2` | `1.4779` | best of the current ReLU²-neighborhood sweep |
| `act4_relu2_ctrl_500` | `relu2` | `1.4837` | strong plain control |
| `act4_swirelu_500` | ReLU + `silu` gate | `1.4854` | close to `relu2`, not better |
| `act5_swiglu_ctrl_500` | `swiglu` | `~1.4833` at step `480` | near `relu2`, not ahead so far |
| `act5_silu2_500` | `silu2` | `1.4956` | smooth single-path square is weaker |
| `act4_relu2_softplus_gate_500` | ReLU² + softplus gate | `1.5123` | smooth positive gate hurts |
| `act5_reluglu_500` | ReLU + linear gate | `1.5086` at step `400` | plain gate is not enough |
| `act4_swirelu2_500` | ReLU² + `silu` gate | `1.5276` | too much gate + square interaction |
| `act5_relu2_lingate_500` | ReLU² + linear gate | `1.5638` at step `400` | unstable / weak |
| `act4_swiglu2_500` | squared SwiGLU | `1.6055` | clearly bad in this setup |

## Is Baseline Still Strongest?

Yes.

The dedicated `500`-step baseline reference is still strongest at `1.4522`.
None of the current activation variants beat it yet.

Within the activation-only sweep, the current ordering is:

1. `gated_relu2`
2. `relu2`
3. `swirelu` / `swiglu` neighborhood
4. smoother or heavier gated-square variants

## What The Parts Seem To Do

### ReLU part

ReLU's hard thresholding still looks important.

Evidence:

- `relu2` is stronger than smoother squared alternatives like `silu2`
- ReLU-based gated variants stay competitive, while fully smooth squared-gated variants fall off

Interpretation:

- hard selection / sparsity seems useful
- current evidence is suggestive, not final

Best direct test:

- compare `relu2` vs `silu2` at the same setup
- compare `swirelu` vs `swiglu`

Why these isolate the question:

- `relu2` vs `silu2` changes the value-path thresholding from hard to smooth while keeping the "square a single path" idea
- `swirelu` vs `swiglu` changes the value path from hard-ReLU to smooth-`silu` while keeping the same smooth gate

If the hard-thresholded version wins in both pairs, that is much better evidence that
hard selection itself matters.

### Square part

The square helps, but not in every placement.

Evidence:

- `relu2` is strong
- `swiglu2` is much worse than both `relu2` and ordinary `swiglu`
- `swirelu2` is worse than `swirelu`

Interpretation:

- "square after ReLU" looks good
- "square inside a heavily gated smooth pathway" looks bad in this setup
- the square is probably useful as amplification of already-selected activations, not as a generic extra nonlinearity

What we know vs do not know:

- we know `swiglu2` performs badly here
- we do not yet know whether that is because of the square itself, because the gate+square interaction is too strong, or because squaring after a smooth value path is the wrong placement

Best direct tests:

- `swiglu` vs `swiglu2`
- `swirelu` vs `swirelu2`
- `relu2` vs `relu2_postsigmoid`

Why these isolate the question:

- `swiglu` vs `swiglu2` asks whether adding the square hurts the standard smooth gated path
- `swirelu` vs `swirelu2` asks the same question but with a ReLU value path
- `relu2` vs `relu2_postsigmoid` asks whether square-before-selection or square-after-selection is the better ordering

### Gate part

A light gate can help, but only if it does not overpower the ReLU² core.

Evidence:

- `gated_relu2` is currently the best activation variant
- `swirelu` stays close to `relu2`
- linear-gate and heavy gate+square combinations are worse

Interpretation:

- a mild learned gate can refine ReLU²
- too much gating freedom seems to blur or destabilize the useful sparse-selection effect

### SwiGLU part

The good part of SwiGLU appears to be the smooth multiplicative gate, not the full package.

Evidence:

- `swirelu` is competitive
- `swiglu` itself looks roughly in the same neighborhood as `relu2`, not clearly better
- `swiglu2` collapses badly

Interpretation:

- borrowing the smooth gate can make sense
- replacing the ReLU² value path with the full SwiGLU style does not look like a win here

## Next Isolation Tests

The cleanest next comparisons are:

1. `relu2` vs `silu2`
2. `swirelu` vs `swiglu`
3. `swiglu` vs `swiglu2`
4. `swirelu` vs `swirelu2`
5. `relu2` vs `relu2_postsigmoid`

These five pairwise checks would answer most of the open "which ingredient matters?"
questions without turning the sweep into random exploration.

## Working Hypothesis

The current best explanation is:

- ReLU contributes sparse hard selection
- squaring amplifies the surviving features
- a small gate can help choose among those features
- but smooth/gated machinery should stay subordinate to the ReLU² core

In short:

`relu2` looks good because it first selects hard, then amplifies.
The most promising modification is not "become SwiGLU".
It is "keep ReLU², add only a light gate".
