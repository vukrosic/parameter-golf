# Parameter Golf Research Pipeline

**Date:** 2026-03-22
**Purpose:** Define a structured, non-indefinite research workflow with clear stage gates, debate triggers, X post cadence, and scaling strategy.

---

## Core Problem

Experiments run in waves indefinitely. There's no stopping condition, no systematic way to decide "this direction is done," and no cadence for external communication. This plan fixes that.

---

## Research Modes

### Mode A: Explore (Broad → Narrow)
Use when starting a new idea category or when stuck on a plateau.
- **Goal:** Screen 4-8 ideas fast, kill losers, find 1-2 worth deepening.
- **Trigger:** New idea from KNOWLEDGE.md, failed direction, or plateau.
- **Process:** See Stage 1 below.

### Mode B: Deepen (Single Direction)
Use when one idea has survived Explore with clear evidence.
- **Goal:** Extract maximum BPB from one winning direction.
- **Trigger:** An Explore winner with ≥0.01 BPB improvement at 500 steps.
- **Process:** See Stage 2 → Stage 3 → Decision.

---

## The Three Stages

### Stage 1 — Quick Screen (500 steps, ~25 min)

**Purpose:** Kill bad ideas cheaply. Find the 1-2 most promising configs.

**How many:** 4-8 experiments per idea category. Run concurrently.

**Queue naming:** `explore/<idea>_<variant>_<n>` e.g. `explore/qat_pct70_500`

**Advancement criteria:**
- Must beat baseline (1.4793 at 500 steps) by ≥0.005 BPB
- Must fit under 16 MB
- Must be reproducible (run at least 2 seeds at 500 steps if result is close to threshold)

**Kill criteria:**
- No improvement at 500 steps → drop direction entirely
- Improvement < 0.005 → too small, drop unless zero-cost trick
- Size violation → redesign (smaller dim, fewer layers) or drop

**X Post after Stage 1:**
```
🚨 Parameter Golf Experiment Results

Quick screen (500 steps, n=4-8 ideas)

🎯 Best: <name> — <val_bpb> (Δ<delta> BPB vs baseline)

💀 Dropped: <list of failed ideas>

📊 Next: <winning idea> → 2000-step validation

#ParameterGolf #ML
```

---

### Stage 2 — Validate (2000-5000 steps, ~2-4 hr)

**Purpose:** Confirm that Stage 1 winners are real and not noise.

**How many:** 1-3 winning ideas × 2 seeds each.

**Queue naming:** `validate/<idea>_<variant>_s<seed>_<steps>` e.g. `validate/moe4e_bn128u_s1337_4k`

**Steps logic:**
- 2000 steps for quick confirmation
- 4000 steps for final validation (preferred)
- 5000 steps only if learning curve is still improving sharply

**Advancement criteria:**
- Must beat baseline (1.4793 at 500 steps scales differently — compare against known curves)
- For MoE4e d384: must beat 1.3637 (current best at 4000 steps)
- Improvement must survive 2+ seeds (Δ < 0.005 between seeds = robust)
- Must fit under 16 MB

**Kill criteria:**
- Result doesn't replicate across seeds → noise, drop
- Improvement shrinks at longer steps (plateauing early) → likely won't win at full run
- Improvement vanishes at 4000 steps when it was visible at 500 → investigate or drop

**X Post after Stage 2:**
```
📈 Parameter Golf — Validation Results

<idea> at <steps> steps (n=2 seeds)

Seed 1337: <val_bpb>
Seed 42:   <val_bpb>
Mean:      <val_bpb>

vs best known: <compare> (Δ<delta> BPB)
vs baseline:  <compare> (Δ<delta> BPB)

✅ Promising — scaling to full run
❌ Dead end — <reason>

#ParameterGolf #ML
```

**Include:** learning curve plot (train loss + val bpb over steps).

---

### Stage 3 — Full Run (13,780 steps, ~12 hr)

**Purpose:** Pre-submission run to confirm the result holds at competition-scale training.

**How many:** 1 config × 2 seeds (only after Stage 2 winner is clear).

**Queue naming:** `full/<idea>_<variant>_s<seed>` e.g. `full/moe4e_bn128u_s1337`

**Advancement criteria:**
- Must beat 1.2244 BPB (current leaderboard baseline) — this is the hard bar
- If within 0.01 BPB of leaderboard baseline, still worth submitting

**Kill criteria:**
- Full run result is worse than Stage 2 trajectory predicted → investigate (likely undertrained or overfitting)
- Budget exhausted → submit best result regardless

**X Post after Stage 3:**
```
🏁 Parameter Golf — Full Run Complete

Config: <model description>
Steps: 13,780 | Wall: ~12hr

Pre-quant:  <bpb>
Post-quant: <bpb>

Leaderboard baseline: 1.2244
Our result:           <bpb> (Δ<delta>)

Status: <submitted / pending / needs more work>

#ParameterGolf #ML
```

---

## The Scaling / Pivot Decision

After Stage 2, you have three choices:

```
IF stage2_result > baseline_by_0.02+ BPB:
    → Scale to full run (Stage 3)
    → Also try stacking with known-good techniques (e.g., bn128_untied + MoE4e)
ELIF stage2_result > baseline_by_0.005 BPB:
    → Try one more Stage 2 variant (hyperparameter sweep)
    → If still promising → Stage 3
ELSE:
    → Pivot. Archive this direction in KNOWLEDGE.md as "falsified" or "insufficient"
    → Return to Stage 1 with a new idea
```

**Maximum depth per direction:** 2 Stage 2 rounds + 1 Stage 3. If it hasn't won by then, archive and pivot.

---

## Debate Agents — When and What

### The Three Agents

**Architect** — Focuses on parameter budget, size constraints, and architectural coherence.
- Asks: "Does this fit under 16MB?" "Are we losing too much capacity to X?"
- Defends: Why the current best config is near-optimal.

**Challenger** — Focuses on what could go wrong at full scale, what might not replicate, and what alternative approaches dominate.
- Asks: "Will this hold at 13k steps?" "Is MoE actually helping or just adding params?" "What if we tried X instead?"
- Pushes: For hard cutoffs and kill criteria.

**Explorer** — Focuses on unexplored directions and wild ideas.
- Asks: "What haven't we tested?" "Is there a orthogonal improvement?"
- Suggests: New experiments to run in parallel.

### When to Trigger Debate

| Event | Trigger? |
|---|---|
| Before starting a new Wave (creating a new queue) | **Always** |
| Stage 1 → Stage 2 transition | **Always** |
| Stage 2 winner identified | **Always** |
| Plateau for 3+ consecutive waves | **Always** |
| Budget below $10 remaining | **Always** |
| Ad-hoc / anytime you feel stuck | Optional |

### What They Debate

The debate output is a **Wave Plan** document written to `queues/wave_NN_plan.md`:

```
# Wave NN — <Topic>

## Decision: <What we're deciding>

## Architect's Position
[150 words max]

## Challenger's Position
[150 words max]

## Explorer's Position
[150 words max]

## Consensus
[One paragraph: what we're doing and why]

## Experiments
<list of queue entries>

## Kill Criteria
<specific conditions that stop this wave early>

## Stage Gate
<what Stage 2 looks like if winners emerge>
```

### Debate Execution

Use Claude Code with three parallel agents, each given a persona prompt + full context (KNOWLEDGE.md, current results, budget status). The Architect, Challenger, and Explorer each produce their section, then a Synthesizer agent (or the human) writes the consensus.

---

## Research vs Deepen: Decision Tree

```
Start of session:
├── What's the current active wave status?
│   ├── If running → check progress, don't start new debate
│   └── If complete → go to Decision Point
│
Decision Point:
├── Did last wave produce a clear winner?
│   ├── YES → Is improvement ≥ 0.01 BPB?
│   │         ├── YES → Mode B: Deepen (Stage 2/3 path)
│   │         └── NO  → Mode B but with more variants (Stage 2 first)
│   └── NO (mixed or negative) → Mode A: Explore new direction
│
Stuck Check:
├── Have we done 3 consecutive waves with no improvement?
│   ├── YES → Emergency debate: is the current approach saturated?
│   │         ├── YES → Pivot to new idea category
│   │         └── NO  → One more focused attempt
│   └── NO → Continue current mode
```

---

## Queue Naming Convention

| Stage | Format | Example |
|---|---|---|
| Explore | `explore/<idea>_<variant>` | `explore/qat_pct70` |
| Validate | `validate/<idea>_<variant>_s<seed>_<steps>` | `validate/moe4e_bn128u_s1337_4k` |
| Full | `full/<idea>_<variant>_s<seed>` | `full/moe4e_bn128u_s1337` |

Archive all completed queues to `queues/archive/wave_NN/<original_file>`.

---

## Budget Allocation Per Stage

Given ~$40 total budget:

| Stage | Max Cost | Rationale |
|---|---|---|
| Explore (per idea) | $3 | 4-8 × 500-step runs on 1 GPU |
| Validate (per winner) | $8 | 2 seeds × 4000 steps |
| Full run | $15 | 2 seeds × 13,780 steps |
| **Total if 1 idea → full** | **~$26** | Leaves buffer |
| **Exploration buffer** | **~$14** | Can try 4-5 ideas before committing |

**Rule:** Never spend >50% of remaining budget on a single Stage 3 run.

---

## X/Twitter Post Cadence

| Milestone | Post? | Content |
|---|---|---|
| Stage 1 complete | **Yes** | Quick screen summary, drop list, next step |
| Stage 2 complete | **Yes** | Validation results, learning curves, advance/drop decision |
| Stage 3 complete | **Yes** | Final result, leaderboard comparison |
| Plateau reached | **Optional** | What we're stuck on, asking for ideas |
| Budget milestone | **Optional** | e.g., "75% of budget used, 3 directions screened" |

**Posting rule:** Posts happen after debate consensus, not spontaneously. The debate output is the post source material.

---

## Skills and Workflow Structure

### New Skill: `/research`

A skill that runs the decision tree above and produces a Wave Plan.

```
/research                    → Full decision + debate trigger
/research mode=explore       → Start new exploration wave
/research mode=deepen        → Continue winning direction
/research mode=debate        → Just run the debate agents
/research mode=post          → Draft X post from last results
```

### New Workflow: `queues/` structure

```
queues/
  active/               ← Current working queue (symlink or text file)
  archive/
    wave_23/
    wave_24/
    ...
  templates/
    explore.txt        ← Template for explore queue entries
    validate.txt        ← Template for validate queue entries
    full.txt            ← Template for full run entries
  wave_NN_plan.md      ← Created by debate, consumed by deploy
```

### File Lifecycle

```
Idea in KNOWLEDGE.md
    ↓ [debate triggered]
queues/wave_NN_plan.md created
    ↓ [queue written]
queues/active.txt = queues/wave_NN.txt
    ↓ [experiments run]
queues/wave_NN.txt → queues/archive/wave_NN/
    ↓ [results analyzed]
KNOWLEDGE.md updated
    ↓ [next debate triggered]
```

---

## What Goes in KNOWLEDGE.md

After each wave, update KNOWLEDGE.md:
- New proven facts (with exact BPB deltas)
- Falsified hypotheses (with evidence)
- Updated ranking of techniques
- Open questions (what we still don't know)

Format per entry:
```
## [Finding name]

**Evidence:** <experiment names>, <step counts>, <exact numbers>
**Delta:** ±X.XXXX BPB vs baseline
**Status:** ✅ proven / ❌ falsified / 🔄 inconclusive
**Notes:** <anything surprising or caveats>
```

---

## Summary: One-Page Process

```
DECIDE          → /research → debate → wave_NN_plan.md
QUEUE           → write queue, archive old active.txt
RUN             → /deploy queues/active.txt
COLLECT         → /collect after completion
ANALYZE         → Stage 1? → compare to 500-step baseline
                  Stage 2? → compare learning curves, seed stability
                  Stage 3? → compare to 1.2244 leaderboard
POST            → X post from debate output
DECIDE          → advance / pivot / deepen
REPEAT          → until budget exhausted or baseline beaten
```

---

## Anti-Patterns to Avoid

1. **Running more Explore without debate** — wastes budget on directions already falsified
2. **Skipping seeds** — single-seed results are noise at this scale
3. **Going to Stage 3 without Stage 2** — full runs cost 10x; confirm first
4. **No post after Stage 2** — forces clarity on whether direction is dead
5. **Keeping a dead queue active** — if a wave is done, archive it immediately
6. **Running more than 2 full directions at once** — budget discipline matters
