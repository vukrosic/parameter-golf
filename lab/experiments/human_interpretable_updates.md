# Activation Research — Status

## What is certain

1. **Squaring the MLP activation gives 0.01–0.05 BPB improvement** for simple activations (relu, identity). Replicated across seeds, large effect. p=2 is optimal.
2. **Squaring destroys gated activations** (SwiGLU diverged). Double multiplicative interaction is unstable.
3. **The pre-squaring function barely matters.** At 2000 steps, all squared variants (relu², abs², leaky², selu², softshrink²) are within 0.005 of each other — mostly within seed noise.
4. **abs² (= x², no activation at all) is the simplest and tied for best.** Mean 1.4698 vs relu² mean 1.4805 at 500 steps (3 seeds). Gap shrinks to ~0.003 at 2000 steps.
5. **Width > gating at matched params.** Full-width relu² beats gated variants that split params between value and gate paths.
6. **Output magnitude alone is useless.** 2·relu(x) without squaring = worse than relu².

## What is uncertain

1. **Does relu² catch up to abs² at long training?** Gap shrinks steadily (0.011 at 500 → 0.003 at 2000 steps). Could converge to zero or plateau at ~0.002. → Wave 24 (6000 steps, running now) resolves this.
2. **Why does squaring work — gradient scaling or expressivity?** The detach experiment (wave 23) was flawed: it only halved gradient scale, both variants still had adaptive gradients. → Wave 25 (`relu2_constgrad`, constant backward gradient) will properly test this.
3. **Is the leak rate sweep confounded by init scale?** relu² has ~half the output variance of abs² at init. The "more leak = better early, less leak = better late" trend could just be optimizer adapting to scale mismatch. → Wave 25 (`relu2_initscale`, output ×2) tests this.
4. **Is "simpler pre-squaring = better" the right framing?** Could be "more signal preservation = better" instead. → Wave 25 (`tanh2`) distinguishes: tanh is simple but compresses signal.

## Running now

- **Wave 24** (6000 steps, GPUs 0/3/4): relu², abs², leaky(0.5)². ~60 min remaining. Checkpoints every 500 steps for learning curve analysis. Will answer: does relu² overtake? [maybe you don't need so many checkpoints, how much memory are they taking vs how much is available? maybe you just need checkpoint at the end]
- **act24_relu2_13000** (GPU 1): 13k-step relu² baseline. ~3 hrs remaining. Full-scale convergence test.

## Next: Wave 25 (1300 steps each, launches when wave 24 finishes)

| Experiment | What it tests | If result is X, then... |
|---|---|---|
| `relu2_initscale` (relu²×2) | Init scale confound | If early gap vs abs² disappears → init scale was the confound, "regularization from zeros" narrative is wrong |
| `relu2_constgrad` (constant backward) | Adaptive gradient vs expressivity | If matches relu² → expressivity alone explains squaring. If much worse → gradient scaling genuinely matters |
| `tanh2` (tanh(x)²) | "Simplicity" vs "signal preservation" | If bad → signal preservation is the right framing. If OK → simplicity was right |

---

## Log

**08:30** — Wrote critique of findings (activation_critique.md). Two high-severity flaws: broken mechanism decomposition, untested init-scale confound. Designed wave 25 to fix both.

**08:00** — Leak rate sweep: at 500 steps more leak = better; at 2000 steps leaky(0.5)² wins. Interpretation uncertain (sparsity-vs-training-length OR init-scale confound).

**07:45** — Wave 24 launched. GPU 2 dead, 4 GPUs active.

**07:30** — Wave 23 mechanism tests complete. Magnitude alone rejected. Detach test showed gradient scale matters but experiment was flawed (see critique).

**06:50** — 2000-step validation: all squared variants within 0.005. relu² worst but closing gap.

**06:30** — Baseline resolved: relu² mean=1.4805 (3 seeds). Original 1.4522 was dirty worktree.
