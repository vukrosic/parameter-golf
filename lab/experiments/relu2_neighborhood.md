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

To understand how to surpass `relu2`, do two things:

1. isolate ingredients at `500` steps
2. multi-seed + extend only the top 2-3 candidates

## Next Tests

Best isolation pairs:

1. `relu2` vs `silu2`
2. `swirelu` vs `swiglu`
3. `swiglu` vs `swiglu2`
4. `swirelu` vs `swirelu2`
5. `relu2` vs `relu2_postsigmoid`

Practical next runs:

1. `relu` vs `silu` to test hard thresholding without the square
2. another seed for `gated_relu2`, `reluglu`, and `swirelu`
3. `1000`-step extensions for the best 2-3 activation candidates

## Working Hypothesis

`relu2` works because it:

1. selects hard
2. then amplifies what survives

Best modification so far:

- keep the ReLU² core
- add only a light gate
