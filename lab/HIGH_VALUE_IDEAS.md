# High-Value Ideas — Prioritized Next Moves

**Context**: We've confirmed relu^2 as the activation winner, found 8/8 heads as a durable architectural improvement, identified MATRIX_LR=0.06 as optimal under warmdown, and have a depth/width ridge sweep in progress. This doc ranks the remaining ideas by expected impact, with reasoning.

---

## 1. Quantization-Aware Training (QAT)

**Why this is the single best bet:**

- The quant gap is 0.0325 BPB on the 4h run — **6.5x the 0.005 target improvement**. Even recovering 15% of that gap wins.
- Zero extra parameters, zero architectural risk. It's purely a training procedure change.
- Orthogonal to every other improvement (architecture, LR, activation). It stacks on top of whatever else we find.
- The 500-step quant gap is only 0.006, so the gap grows with training length. At the full 13,780-step submission run, the gap will be substantial — and that's exactly where QAT pays off.
- Implementation is straightforward: fake-quant via STE in CastedLinear.forward(), enable in last 20-30% of steps.

**Risk**: Low. Worst case it doesn't help and we remove it. STE is well-understood.

**Prerequisite**: Measure the actual quant gap at 2000+ steps on current best config (8/8 heads, LR=0.06). If the gap is <0.002 at that horizon, QAT is less urgent.

**Expected ROI**: 0.005-0.015 BPB improvement (recoverable portion of quant gap).

---

## 2. Shared-Weight Layer Cycling

**Why this is high-value:**

- Our depth/width sweep is already showing 12x448 > 9x512. Depth helps. This idea gets more effective depth for free parameters.
- 5 unique layers x 2 cycles = 10 effective layers, but only 5 layers of parameters. The freed parameter budget can be reinvested into width (e.g., dim=640).
- The 600s compute budget is generous — doubling FLOPs still leaves ~7,000 steps, which is enough for convergence.
- Weight sharing acts as implicit regularization, which may help at this tiny scale where overfitting to weight noise is a concern (see: all our quant gap issues).
- Per-cycle layer norms or learned scalars break symmetry cheaply.

**Risk**: Medium. U-Net skip connections break under cycling (encoder/decoder split becomes ambiguous). Need to either disable skips or redesign them. Also, Muon optimizer sees gradients from all cycle positions — may need LR/num_cycles adjustment.

**Key experiment**: 5 blocks x 2 cycles vs 9-block baseline at 500 steps, with U-Net skips disabled in both for fair comparison.

**Expected ROI**: 0.005-0.020 BPB if depth is truly the bottleneck (early evidence says yes).

---

## 3. Factorized Embeddings

**Why this matters:**

- The embedding table is 1024x512 = 524K params. With only 1024 tokens, a 512-dim embedding is almost certainly overparameterized. A 64-dim bottleneck (1024x64 + 64x512 = 98K params) saves ~426K params = ~400KB under int8.
- 400KB is enough for roughly +1 transformer layer or +24 dims of width. Given that depth helps, reinvesting into an extra layer is attractive.
- This interacts well with shared-weight cycling: freed params from embeddings + freed params from weight sharing = significantly wider or deeper model within the same 16MB cap.
- The factored embedding may actually quantize better (two small matrices with constrained rank vs one large one).

**Risk**: Low-medium. The tied output projection becomes a two-step factored projection, which might slightly hurt output logit quality. Easy to test: compare bottleneck=64 vs 128 vs full.

**Expected ROI**: 0.003-0.010 BPB from reinvested capacity.

---

## 4. Depthwise Conv Prefix

**Why it's worth trying:**

- Only ~14K extra params (kernel=3) across 9 layers. Negligible parameter cost.
- Gives each layer a cheap local receptive field, freeing attention heads to specialize on longer-range dependencies. With only 8 heads, each head doing double duty (local + global) is wasteful.
- Mamba/H3 work shows local conv is complementary to attention, not redundant.
- Implementation is simple: one causal conv1d per block before attention.

**Risk**: Low. Worst case it adds 14K params for no benefit. May marginally slow step time.

**Expected ROI**: 0.002-0.008 BPB. Modest but nearly free.

---

## 5. Per-Layer LR Scaling

**Why it's low-hanging fruit:**

- Zero params, zero FLOPs, zero implementation risk. Just multiply gradients by a per-layer scalar.
- The model has 9 layers with very different roles (early = token mixing, late = prediction). A uniform LR is almost certainly suboptimal.
- Can be tested in 2 runs: linear decay (early=1.0, late=0.5) vs inverse (early=0.5, late=1.0).
- Interacts with existing Muon vs Adam LR split — could unlock better training dynamics.

**Risk**: Essentially zero. Two 500-step runs.

**Expected ROI**: 0.001-0.005 BPB. Small but free.

---

## 6. Softmax Temperature Annealing

**Why it's interesting despite being speculative:**

- Zero params, zero FLOPs. Pure schedule change.
- The intuition is sound: early training benefits from soft, exploratory attention; late training benefits from sharp, confident patterns.
- The model already has qk_gain_init=1.5 and logit_softcap=30.0, which are static temperature-like knobs. Annealing adds a dynamic dimension.
- Could compound with the LR warmdown that's already active — as LR decays, attention temperature could sharpen.

**Risk**: Low, but interactions with existing qk_gain and softcap are unpredictable. Need careful ablation.

**Expected ROI**: 0.001-0.005 BPB. Speculative.

---

## Stacking Order

These ideas are mostly orthogonal. The recommended stacking order (test each independently first, then combine winners):

1. **QAT** — pure training trick, stacks on anything
2. **8/8 heads + LR=0.06** — already validated, use as new baseline
3. **Depth/width winner** from current ridge sweep
4. **Factorized embeddings** — frees params for #5
5. **Shared-weight cycling** — uses freed params + more depth
6. **Depthwise conv** — cheap add-on
7. **Per-layer LR** — final training tune
8. **Softmax temp annealing** — final training tune

Each step should be validated at 500-1000 steps before stacking with the next.

---

## What's NOT On This List (and Why)

| Idea | Why deprioritized |
|------|-------------------|
| Sparse MoE | PyTorch dispatch overhead with mask loops is slow without custom kernels. At 2 experts the specialization benefit is unclear. Implementation cost is high relative to expected gain at 17M scale. |
| SSM-Attention Hybrid | Very high implementation cost (Mamba from scratch in 1500 lines). Unclear benefit at seq_len=1024 where attention is already fast. |
| Progressive Dimension Growth | Adapter projections between dimension changes eat the parameter savings. Adds complexity for uncertain payoff. |
| Byte-Level Fallback Head | Clever metric exploitation but the math is unlikely to work out — a 256-way byte head is expensive and BPE-1024 already handles most tokens well. |
| GEGLU/SwiGLU | Tested extensively in activation ablations. 13% slower per step, plateau earlier than relu^2. Dead. |
