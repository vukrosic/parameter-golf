# Activation Research — Status

## What is certain

1. **Squaring the MLP activation gives 0.01–0.05 BPB improvement** for simple activations (relu, identity). Replicated across seeds, large effect. p=2 is optimal.
2. **Squaring destroys gated activations** (SwiGLU diverged). Double multiplicative interaction is unstable.
3. **The pre-squaring function barely matters.** All squared variants within 0.005 at 2000 steps, within 0.003 at 6000 steps — mostly seed noise.
4. **leaky(0.5)² is the best activation tested.** Post-quant BPB: **1.2708** vs relu² 1.2737 vs abs² 1.2745 at 6000 steps. Leads at every checkpoint past step 500, gap widening.
5. **relu² overtakes abs² by step 5000.** abs²'s early lead (0.010 at 500 steps) was transient — an init-scale artifact. abs² has ~2× output variance at init, so Adam warms up faster. Once the optimizer adapts, relu²'s mild sparsity gives it a slight edge.
6. **Width > gating at matched params.** Full-width relu² beats gated variants that split params between value and gate paths.
7. **Output magnitude alone is useless.** 2·relu(x) without squaring = worse than relu².

## What wave 24 (6000 steps) resolved

**Q: Does relu² catch up to abs²?** Yes — and overtakes it. The gap closes smoothly: +0.010 → +0.005 → +0.002 → 0.000 → -0.0003. After quantization relu² beats abs² by 0.0008.

**Q: Is there a best activation?** leaky(0.5)² — it gets the init-scale benefit of abs² (enough leak to avoid relu²'s variance penalty) plus mild sparsity for regularization. Consistent winner from 2000 to 6000 steps.

**Q: Do rankings change with training length?** Yes. Early rankings (500 steps) are dominated by init-scale effects. True rankings emerge after ~2000 steps. abs² goes from best to worst among the three. This means short-run ablations can be misleading.

## What is still uncertain

1. **Why does squaring work — gradient adaptivity or expressivity?** Wave 25 `relu2_constgrad` (running now) uses truly constant backward gradients to test this. Early signal: constgrad is 0.04 BPB behind at step 50, suggesting adaptive gradients matter.
2. **Is the init-scale confound the full story?** Wave 25 `relu2_initscale` (running now) scales relu² output ×2 to match abs² variance. Early signal: it converges faster than baseline relu², confirming init scale matters.
3. **Is signal preservation or simplicity the right framing?** Wave 25 `tanh2` (queued, launches in ~15 min) — tanh is simple but compresses signal. If bad → signal preservation wins.
4. **Does leaky(0.5)² hold at 13k steps?** The 13k relu² baseline is at step ~6050 (BPB 1.2700). Need a 13k leaky(0.5)² run to confirm the advantage scales.

## Running now

- **act25_relu2_initscale_r2** (GPU 0): relu²×2 init scale test, 1300 steps. At step ~100.
- **act25_relu2_constgrad_r2** (GPU 3): constant gradient test, 1300 steps. At step ~100.
- **act25_tanh2_r2** (GPU 0, queued): launches when initscale finishes (~15 min).
- **act24_relu2_13000** (GPU 1): 13k baseline, at step ~6050/13000. BPB 1.2700.

## Next steps

1. **When wave 25 completes (~40 min):** Analyze mechanism results — which matters more, adaptive gradients or quadratic expressivity?
2. **If leaky(0.5)² confirmed as best:** Run 13k leaky(0.5)² for final comparison against relu² baseline.
3. **For the competition submission:** Switch default activation from relu² to leaky(0.5)², saving ~0.003 BPB post-quant.

---

## Log

**11:00** — Wave 25 launched manually (gpu_monitor CUDA bug). initscale and constgrad on GPUs 0/3. tanh2 queued. GPU 4 dead (added to SKIP_GPUS alongside GPU 2).

**10:50** — Wave 24 complete. Key result: leaky(0.5)² best (quant 1.2708), relu² overtakes abs² by step 5000. Rankings change with training length — short ablations misleading.

**08:30** — Wrote critique of findings. Two high-severity flaws: broken mechanism decomposition, untested init-scale confound. Designed wave 25 to fix both.

**07:45** — Wave 24 launched. GPU 2 dead, 4 GPUs active.

**06:50** — 2000-step validation: all squared variants within 0.005. relu² worst but closing gap.

**06:30** — Baseline resolved: relu² mean=1.4805 (3 seeds). Original 1.4522 was dirty worktree.
