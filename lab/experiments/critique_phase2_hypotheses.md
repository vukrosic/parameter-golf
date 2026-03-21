# Critique: Phase 2 Hypotheses and Experiments

## The three hypotheses

- **H1:** Activation-proportional gradients are critical (grad ∝ activation magnitude)
- **H2:** Signal compression hurts (bounding the output range destroys information)
- **H3:** Preserving negative-side information helps (passing signal for x < 0 improves learning)

## What H1 actually showed vs what we claim

**Claim:** "Proportional gradients are the dominant mechanism behind relu²'s advantage."

**What the data says:** Const-grad penalties of +0.11-0.12 BPB are massive and real. No dispute there. But "proportional gradients are critical" is a weaker finding than it sounds. Here's why:

1. **We only tested one extreme.** Const-grad replaces grad=2x with grad=1 everywhere. That's not a surgical test of proportionality — it's a nuclear bomb that also destroys all gradient *variation*. Of course uniform gradients perform terribly: every neuron gets the same update magnitude regardless of its current activation. We proved that *gradient uniformity is catastrophic*, but that's a much weaker statement than "proportionality is optimal."

2. **The gradfloor result undermines pure proportionality.** `gradfloor` (which floors the gradient to a minimum of 0.5) *beat* the natural 2x gradient. If pure proportionality were optimal, any deviation should hurt. Instead, adding a floor helped. This suggests the real story is: **gradients should vary with activation, but zero gradients for small activations are wasteful.** The floor gives near-dead neurons a recovery pathway that pure proportional gradients don't.

3. **The multiplier doesn't matter much.** 1.5x, 2x, and 3x all perform within 0.003 BPB of each other. This means the system is robust to the exact scaling — it's the *shape* (monotonically increasing) that matters, not the slope. The hypothesis as stated ("activation-proportional") overspecifies what the data supports.

**Better framing of H1:** "Gradients must be monotonically increasing with activation magnitude. Uniform gradients are catastrophic. The exact slope doesn't matter, but having a nonzero minimum (floor) may help by preventing neuron death."

**Missing experiment:** We never tested a *decreasing* gradient schedule (e.g., grad = 1/x). That would be the true control for "must be increasing" vs "must vary." Also never tested a random gradient perturbation to see if it's the deterministic relationship that matters or just the variance.

---

## H2: Signal compression — early results are alarming

**Intermediate H2 results (step ~200/500, not final):**

| Experiment | val_bpb @~200 | vs relu² @200 (~1.56) |
|---|---:|---|
| `erf²` | 1.6736 | +0.11 |
| `hardtanh²` | 1.6885 | +0.13 |
| `clamped16` | 1.6910 | +0.13 |
| `sigmoid²` | 1.7580 | +0.20 |
| `clamped4` | 1.7916 | +0.23 |
| `tanh_scaled²` | 1.8005 | +0.24 |
| `softsign²` | 1.8152 | +0.26 |
| `arctan²` | 2.6908 | +1.13 (!) |

**Critique of H2 design:**

1. **These experiments conflate two things.** When we say "signal compression hurts," we're mixing up (a) output range bounding and (b) gradient saturation. For tanh², sigmoid², etc., the gradient goes to zero for large inputs (tanh'(x) → 0 as |x| → ∞). So the network can't update neurons that have large pre-activation values. Is it the *bounding* that hurts, or the *gradient death*? These are different mechanisms with different implications.

   - `clamped16` is the test for this: it has relu²'s natural gradient for |x| < 4 but kills gradient beyond that. The fact that clamped16 is already +0.13 BPB worse at step 200 suggests that **even rare large activations carry important gradient signal** — or alternatively, that gradient death at the clamp boundary creates optimization pathology.

2. **The arctan² result (2.69 BPB!) suggests something beyond compression is going wrong.** arctan(x)² has a bounded range similar to tanh², so we'd expect similar penalty. But it's catastrophically worse — 1.13 BPB above baseline. Likely explanation: arctan has a different gradient profile that interacts badly with the optimizer. arctan'(x) = 1/(1+x²), which decays much more slowly than tanh'(x), so gradient saturation alone doesn't explain it. There may be a numerical issue or the specific output range [0, π²/4] ≈ [0, 2.47] creates a scale mismatch. **This suggests our H2 framing is too simple** — it's not just "bounded vs unbounded" but the entire interaction of output scale, gradient shape, and optimizer dynamics.

3. **H2 may be a consequence of H1, not an independent mechanism.** Consider: bounded activations have saturating gradients (grad → 0 for large |x|). This violates H1 (gradients should scale with activation). So every H2 experiment that fails might be failing because of H1 violation, not because of the bounding per se. The experiments don't disentangle this. **Key missing experiment:** A bounded activation with artificially maintained proportional gradients (custom backward that keeps grad ∝ x even inside tanh). If that recovers performance, H2 is just H1 in disguise.

4. **`log1p_relu²` is the crucial discriminator and it hasn't run yet.** log1p(relu²(x)) is unbounded but compresses growth. If it's fine, H2 is about gradient saturation (= H1). If it also fails, growth rate compression is independently bad.

---

## H3: Negative-side information — the weakest hypothesis

**Critique:**

1. **The evidence for H3 is thin.** The entire case rests on leaky(0.5)² being ~0.003 BPB better than relu². That's at the noise floor (~0.003 seed variance). Yes, it's consistent across checkpoints. Yes, all three leak rates (0.3, 0.5, 0.7) beat relu². But:
   - We have 2 seeds, not 10. Consistent-across-checkpoints doesn't mean consistent-across-seeds.
   - The leak sweep used mismatched seeds (0.5 used seed 42, 0.3/0.7 used seed 1337), making within-sweep comparisons unreliable.

2. **H3 is confounded with H1.** leaky(0.5)² doesn't just add negative-side information — it also changes the gradient landscape for negative inputs. relu² has grad=0 for x<0 (complete gradient death). leaky(0.5)² has grad=x for x<0 (proportional gradient). So leaky winning could be entirely explained by H1: "more neurons with nonzero gradient = better." The `relu²_linneg0.5` experiment (linear negative, constant grad) is designed to test this, but it hasn't run yet. If linneg0.5 matches leaky(0.5)², then H3 is just H1 for the negative side.

3. **The H3 experiment set has too many similar variants.** We're testing leaky at 0.1, 0.2, 0.5, 0.8 — four points on the same curve. That's fine for mapping a dose-response, but it's 4 experiments to answer one sub-question. Meanwhile, the truly interesting H3 tests (`x_absx`, `bipolar_relu²`, `relu²_linneg0.5`) each test genuinely different mechanisms. The set is overweight on leaky variants and underweight on mechanistic controls.

4. **`x_absx` (sign(x)·x²) is the most important single experiment in the whole Phase 2 queue.** It has the same gradient magnitude as relu² (2|x|) everywhere — perfectly satisfying H1 — but preserves sign. If x·|x| beats relu², H3 is real and independent of H1. If it ties or loses, the negative-side advantage was always about gradient recovery (H1), not information (H3). Every other H3 experiment is secondary to this one.

---

## Structural critique: are these even the right questions?

1. **The goal is to beat 1.2244 BPB, not to understand activation functions.** Understanding is useful insofar as it guides search, but we're spending a lot of GPU time on scientific questions (dose-response curves, mechanism isolation) that won't directly produce a better activation. The 500-step screening budget would be better spent on:
   - Running the top 3 candidates (leaky(0.5)², selu², x·|x|) at 2000+ steps immediately
   - Testing interactions with other hyperparameters (LR, warmup, architecture)
   - Trying completely novel shapes informed by the mechanistic understanding, rather than exhaustively characterizing known shapes

2. **The hypotheses are not independent.** H1 (gradient scaling), H2 (signal compression), and H3 (negative information) interact heavily. Bounded activations violate H1 via gradient saturation. Negative-side activations satisfy H1 for more neurons. This means the 3-hypothesis framework creates an illusion of orthogonal dimensions when the underlying space may be 1-dimensional: "what fraction of neurons have useful gradients at any given time?"

   **Alternative single-hypothesis framing:** *The optimal activation maximizes the fraction of neurons that receive meaningful, differentiated gradient updates at each step.* This subsumes all three hypotheses:
   - H1: proportional gradients differentiate neurons → ✓
   - H2: saturating gradients kill differentiation for large activations → ✓ (special case)
   - H3: negative-side gradients increase the fraction with nonzero updates → ✓ (special case)

   This framing predicts that `gradfloor` should help (it did — gives minimum gradient to near-dead neurons) and that `x_absx` should be the optimal activation (every neuron always has nonzero, proportional gradients).

3. **We're not testing the quantization interaction.** The competition metric is *post-quantization* BPB. An activation that produces activations with smaller dynamic range might quantize better even if it's slightly worse pre-quant. We're not testing this at all in the 500-step screen (quant eval only runs at the end). An activation that's 0.002 BPB worse but has 0.001 smaller quant gap would win. The clamped variants might actually be interesting for this reason — clamping reduces dynamic range.

4. **500-step screening has known failure modes.** The findings document itself acknowledges that abs² led early but tied relu² by step 5000. We've already been burned by this. Yet we're making elimination decisions at 500 steps for 40+ activations. Any activation within ~0.01 BPB of the baseline at 500 steps could be the actual winner at 13k steps. We should be more conservative about elimination.

---

## Specific experiment critiques

**Experiments that should be prioritized (high information value):**
- `x_absx` — cleanest H3 test, controls for H1 perfectly
- `log1p_relu²` — disentangles H2 from H1 (is it bounding or gradient saturation?)
- `relu²_linneg0.5` — disentangles H3 from H1 (is it information or gradient recovery?)
- `selu²` rerun — if real, it satisfies all three hypotheses, making it the best candidate

**Experiments with lower information value:**
- `leaky(0.1/0.2/0.8)²` — dose-response is nice but we already know leaky works. 3 experiments for marginal information.
- `gelu²` and `mish²` — "where does the popular activation rank?" is benchmarking, not hypothesis testing. Fine to include but low priority.
- `softplus²_beta5/0.5` — curvature at origin is a second-order effect. Unlikely to produce actionable findings.

**Experiments that are missing:**
- **Bounded activation + proportional custom gradient** — the definitive H1 vs H2 separation test
- **x_absx with different negative scales** — e.g., sign(x)·|x|^1.5 or 0.5·sign(x)·x² — to find the optimal negative-to-positive ratio if H3 is real
- **Activation + LR co-optimization** — different activations may have different optimal LR. We removed the LR variants as "not hypothesis testing" but this matters for fair comparison

---

## Summary of weaknesses

| Hypothesis | Status | Confidence | Main weakness |
|---|---|---|---|
| H1 | Confirmed | High for "uniform is bad", Medium for "proportional is optimal" | Only tested one extreme (const-grad). gradfloor beating baseline undermines pure proportionality. |
| H2 | Testing | Preliminary results support it | Likely not independent from H1 (bounded → gradient saturation → H1 violation). arctan² anomaly unexplained. |
| H3 | Untested (only legacy data) | Low | Signal is at noise floor. Confounded with H1. Key experiment (x_absx) hasn't run yet. |

**The single most important thing to learn from Phase 2:** Does `x_absx` beat relu²? If yes, H3 is real. If no, collapse H1+H2+H3 into "maximize useful gradient coverage" and focus on gradfloor-style modifications.
