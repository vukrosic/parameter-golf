# ReLU2 Neighborhood Sweep

## Read This

- `Current Results`
- `What Seems True`
- `Next Tests`

## Skip If You Want

- `Run Set`
- `Phase 2`

## Goal

Figure out which part of `relu2` helps:

- hard ReLU selection
- squaring
- no gate

and which part of SwiGLU is worth borrowing:

- smooth multiplicative gate

Rules for this lane:

- causal isolation tests only
- ingredient-comparison pairs
- no extra-seed confirmation yet
- no `1000`-step extension yet

## Run Set

Wave 1:

- `act4_relu2_ctrl_500`
- `act4_gated_relu2_500`
- `act4_swirelu_500`
- `act4_swirelu2_500`
- `act4_relu2_softplus_gate_500`
- `act4_swiglu2_500`

Wave 2:

- `act5_swiglu_ctrl_500`
- `act5_silu2_500`
- `act5_reluglu_500`
- `act5_relu2_lingate_500`
- `act5_relu2_postsigmoid_500`

Wave 3:

- `act7_relu_500`
- `act7_silu_500`
- `act7_gelu_500`
- `act7_swiglu_500`
- `act7_swirelu2_500`
- `act7_swiglu2_500`
- `act7_relu2_postsigmoid_500`
- `act7_relu2_lingate_500`

Wave 4:

- `act9_relu15_500`
- `act9_relu22_500`
- `act9_relu25_500`
- `act9_gated_relu22_500`
- `act10_relu18_500`
- `act10_relu24_500`
- `act10_mild_gated_relu2_500`
- `act10_mild_gated_relu22_500`

## Current Results

| Run | BPB | Read |
|---|---:|---|
| `baseline_500` | `1.4522` | still best overall |
| `act5_swiglu_ctrl_500` | `1.4681` | best smooth-gated baseline |
| `act5_reluglu_500` | `1.4733` | ReLU + plain gate is decent |
| `act4_gated_relu2_500` | `1.4779` | best ReLU²-family variant so far |
| `act4_relu2_ctrl_500` | `1.4837` | strong plain control |
| `act5_silu2_500` | `1.4833` | smooth squared path is only okay |
| `act4_swirelu_500` | `1.4854` | smooth gate on ReLU stays close |
| `act5_relu2_lingate_500` | `1.4965` | linear gate on ReLU² is worse |
| `act4_relu2_softplus_gate_500` | `1.5123` | smooth positive gate hurts |
| `act4_swirelu2_500` | `1.5276` | square + smooth gate hurts |
| `act4_swiglu2_500` | `1.6055` | clearly bad |

## What Seems True

- Baseline is still strongest.
- `relu2` is the activation control, not the overall training baseline.
- `baseline_500` is the overall reference run; `relu2` is the default MLP activation inside the activation sweep.
- A light gate can help, but only a little.
- Smooth gating is not obviously bad.
- Heavy gate + square combinations look bad.
- `swiglu2` is bad here.

## What Each Part Seems To Do

ReLU:

- likely helps through hard selection
- evidence: `relu2` and ReLU-gated variants beat smoother square-heavy variants

Best direct tests:

- `relu` vs `silu`
- `relu2` vs `silu2`

Square:

- helps in `relu2`
- hurts when combined with too much smooth gating

Best direct tests:

- `swiglu` vs `swiglu2`
- `swirelu` vs `swirelu2`
- `relu2` vs `relu2_postsigmoid`

Gate:

- a small gate can help
- evidence: `gated_relu2` beats plain `relu2`, and `reluglu` is competitive
- too much gate freedom seems harmful

SwiGLU:

- the useful part may be the smooth gate
- to improve `relu2`, the most relevant tests are ReLU² plus a light gate, not full smooth gated-square stacks
- the full squared SwiGLU path looks bad

To understand how to surpass `relu2`, stay in ingredient-isolation mode first.

## Next Tests

Best isolation pairs:

1. `relu2` vs `silu2`
2. `swirelu` vs `swiglu`
3. `swiglu` vs `swiglu2`
4. `swirelu` vs `swirelu2`
5. `relu2` vs `relu2_postsigmoid`
6. `relu18` vs `relu2` vs `relu22` vs `relu24` vs `relu25`
7. `gated_relu2` vs `mild_gated_relu2`
8. `gated_relu22` vs `mild_gated_relu22`

Practical next runs:

1. `relu` vs `silu` to test hard thresholding without the square
2. `relu` vs `gelu` to compare hard vs softer single-path activations
3. `swiglu` vs `swiglu2` to isolate the effect of adding the square
4. `swirelu` vs `swirelu2` to isolate the same question with a ReLU value path
5. `relu2` vs `relu2_postsigmoid` to test square placement
6. `relu2` vs `relu2_lingate` to test whether a plain gate helps or hurts
7. power-law ReLU around `2.0` to test whether the win is specific to exactly squaring
8. mild gated ReLU² to test whether the gate should refine the core instead of dominating it

## Working Hypothesis

`relu2` works because it:

1. selects hard
2. then amplifies what survives

Best modification so far:

- keep the ReLU² core
- add only a light gate
