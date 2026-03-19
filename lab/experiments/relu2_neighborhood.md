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
