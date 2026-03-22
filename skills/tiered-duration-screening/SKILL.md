---
name: tiered-screen
description: Screen architecture variants with a fast 1s→2s→3s ladder. Promote top runs at each stage. Write all three result tables to a markdown report file.
---

# Tiered Architecture Screen

## What This Does

Runs many architecture variants cheaply, promotes winners to longer runs, drops losers early. Produces a markdown report file with three comparison tables — one per stage.

## Ladder

- Default: `1s → 2s → 3s`, counts `10 → top 5 → top 3`
- Use `MAX_WALLCLOCK_SECONDS` as the duration knob, set `ITERATIONS` high so wallclock is the limit.
- Adjust ladder and counts if the user specifies different durations or promotion counts.

## Steps

### 1. Run Stage 1
Run all candidates at the first duration. Include a baseline run.

### 2. Promote and run Stage 2
Sort by `delta_loss = candidate_loss - baseline_loss` (lower is better). Re-run the top k at the second duration, plus a fresh same-duration baseline.

### 3. Promote and run Stage 3
Same as above: promote top k from Stage 2, run at the third duration with a fresh baseline.

### 4. Write the report file

**Always write results to `results/tiered_screen_<timestamp>.md`** (or a name the user specifies).

The file must contain three markdown tables plus a short conclusion. Never just print to the terminal — write the file.

Table format (one per stage):

```markdown
### Stage 1 — 1s

| Run | Duration | Architecture change | Loss | Baseline loss | Delta loss | Decision |
|---|---:|---|---:|---:|---:|---|
| tiered_1s_baseline | 1s | baseline | 6.9312 | 6.9312 | +0.0000 | baseline |
| tiered_1s_swiglu   | 1s | SwiGLU activation | 6.9280 | 6.9312 | -0.0032 | promote ✓ |
```

Finish with 2–4 sentences: what survived, what died, whether the ladder is stable enough to justify longer validation runs.

## Repo Conventions

- Stage labels: `tiered_1s`, `tiered_2s`, `tiered_3s`
- Per-run results: `results/<run_id>/summary.json`
- Report file: `results/tiered_screen_<timestamp>.md`

## Duration Overrides

Replace the ladder with any durations the user specifies. Keep promotion counts aligned unless told otherwise.
