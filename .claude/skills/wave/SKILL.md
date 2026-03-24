---
name: wave
description: Unified research loop — check status, run debates, approve plans, collect results, draft X posts. Replaces /research with a tighter loop.
---

# Wave — Research Loop Orchestrator

The single entry point for the research pipeline. Each invocation either shows status or advances the loop by one phase.

## Usage

### `/wave` (no args)
Show current wave status and auto-suggest the next action.

**What it does:**
1. Check if experiments are running (`/fleet`)
2. Check if there's an active queue (`queues/active.txt`)
3. Check latest results
4. Determine which phase we're in and what to do next

**Decision logic:**
```
IF experiments running → "Wave N in progress. Use /status to monitor."
IF active queue exists but not deployed → "Queue ready. Use /deploy to start."
IF results ready but not analyzed → "Results in. Run /wave results."
IF no active work → Determine debate type and suggest /wave plan.
```

### `/wave plan`
Run a debate and generate a wave plan.

**What it does:**
1. Auto-detect debate type based on current state:
   - **Direction**: No active winners, need new ideas (uses Architect + Explorer + Skeptic)
   - **Scale**: Explore winners exist, deciding validate/full (uses Architect + Challenger + Optimizer)
   - **Pivot**: 3+ stale waves or budget <$10 (uses all 5 agents)
2. Run `bash debate/debate.sh "<topic>" 3 <debate_type>`
3. Generate `queues/wave_NN_plan.md` and `queues/wave_NN.txt`
4. Display plan for human review

**Override debate type:** `/wave plan type=direction|scale|pivot`
**Override topic:** `/wave plan topic="your specific topic"`

### `/wave approve`
Lock the current wave plan and activate the queue.

**What it does:**
1. Find the latest `queues/wave_NN_plan.md` with status PENDING
2. Show the plan and queue for final review
3. Copy `queues/wave_NN.txt` → `queues/active.txt`
4. Update plan status to APPROVED
5. Prompt: "Ready to deploy. Run: /deploy queues/active.txt"

### `/wave results`
Collect results, compare experiments, and draft an X post.

**What it does:**
1. Run `/collect` to pull results from all GPUs
2. Run `/compare` to rank experiments
3. Determine which phase just completed (explore/validate/full) from experiment names
4. Apply gate criteria:
   - Explore: any winner >0.01 BPB over baseline?
   - Validate: seeds agree within 0.005 BPB?
   - Full: beats 1.2244 BPB?
5. Draft X post appropriate to the phase
6. Archive completed queue to `queues/archive/wave_NN/`
7. Update KNOWLEDGE.md with findings
8. Suggest next action

### `/wave pivot`
Force a pivot debate regardless of current state.

**What it does:**
1. Run a Pivot debate (all 5 agents)
2. Topic: "Strategic reassessment: are we saturated? What's the highest-EV path?"
3. Generate plan as normal
4. This is for when you're stuck and need a fresh perspective

## Phase Detection

The skill auto-detects the current phase by examining:
- `queues/active.txt` content (experiment name prefixes: explore/, validate/, full/)
- Latest results in `results/explore/`, `results/validate/`, `results/full/`
- Wave plan status (PENDING vs APPROVED)
- Fleet status (training vs idle)

## Debate Type Auto-Selection

```
gap = current_best_bpb - 1.2244  (leaderboard target)

IF gap > 0.05:    → Direction (need breakthrough)
IF gap <= 0.05:   → Scale (squeeze winning config)
IF budget < $10:  → Pivot (reassess)
IF 3+ waves no improvement: → Pivot
```

Override with `/wave plan type=direction|scale|pivot`.

## Scripts
- `scripts/wave.sh` — main orchestrator (status, plan, approve, results)

## Key Files Read
- `KNOWLEDGE.md` — proven facts and failed approaches
- `research/PIPELINE.md` — pipeline definition
- `queues/active.txt` — current queue
- `queues/wave_*_plan.md` — wave plans
- `debate/CONCLUSIONS.md` — debate history
- `results/*/summary.json` — experiment results

## Key Files Written
- `queues/wave_NN_plan.md` — new wave plan
- `queues/wave_NN.txt` — new queue file
- `queues/active.txt` — activated queue
- `debate/CONCLUSIONS.md` — appended debate output
