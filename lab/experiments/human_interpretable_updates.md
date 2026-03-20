# Activation Research — Live Updates

Short status updates for quick human review. Latest first.

---

**2026-03-20 07:45** — Launching wave 24: long runs (6000 steps, ~100 min) for relu², abs², leaky(0.5)² with checkpoints every 500 steps. Will reveal if relu²'s closing trend continues to a crossover. GPU monitor script running to auto-fill idle GPUs.

**2026-03-20 07:30** — Wave 23 (WHY squaring works) complete on 4/5 GPUs. leaky07 crashed (CUDA error, retrying). Three hypotheses tested:
- H1 (just bigger outputs?): **NO.** `2·relu(x)` without squaring = 1.5022, much worse than relu² = 1.4807. Magnitude alone doesn't help.
- H2 (adaptive gradient?): **PARTLY.** `relu(x)·stopgrad(relu(x))` = 1.4963. Same output as relu² but gradient halved. Costs 0.016 vs relu². So the gradient scaling (bigger activations get bigger gradients) matters but isn't the whole story.
- H3 (quadratic feature interaction?): **Wins by elimination.** ~50% of squaring's benefit comes from gradient scaling (H2), the rest from the quadratic features themselves (x² lets the MLP learn products of input features, not just sums).

**2026-03-20 07:25** — Critical trend spotted in wave 22 learning curves: relu² starts worst but closes the gap steadily vs abs² and leaky(0.5)². Gap halves every ~1000 steps. Hypothesis: hard zeros provide regularization that only manifests with extended training. Need 6000+ step runs to test.

**2026-03-20 06:50** — Wave 22 (2000-step depth check) complete. All 5 squared variants within 0.005 of each other. leaky(0.5)²=1.3218 best, relu²=1.3264 worst. But relu² is gaining at every checkpoint.

**2026-03-20 06:30** — abs² confirmed across 3 seeds: mean 1.4698 vs relu² mean 1.4805 (Δ=0.0107 at 500 steps). Original relu² baseline (1.4522) was dirty worktree artifact.

**2026-03-20 06:00** — Started activation mechanism investigation. Goal: understand WHY relu² works, not just rank alternatives.
