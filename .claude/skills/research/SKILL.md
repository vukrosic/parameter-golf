---
name: research
description: Orchestrate the research pipeline — run debate agents, generate wave plans, create queues, manage the explore→validate→full workflow.
---

# Research Pipeline Orchestration

## Modes

### `/research` (no args)
Run the full decision loop: check status → determine mode → trigger debate → generate wave plan → prompt to deploy.

### `/research mode=decide`
Read current results, check active waves, consult KNOWLEDGE.md, and output:
- Current wave status (running / complete / none)
- Decision: Explore new direction OR Deepen winning direction
- Why that decision was made

### `/research mode=debate [topic]`
Run the 3-agent debate (Architect / Skeptic / Explorer) on the given topic or auto-detected topic. Outputs to `debate/rounds/<timestamp>/`. After debate, generates `queues/wave_NN_plan.md`.

### `/research mode=queue`
Generate a queue file from the most recent wave plan. Ask user to confirm before writing to `queues/active.txt`.

### `/research mode=deepen`
Called when a Stage 1 winner exists. Triggers debate focused on deepening the winning direction (hyperparameter sweeps, stacking, seed replication).

### `/research mode=explore`
Called when no clear winner exists. Triggers debate focused on exploring a new idea category.

## Wave Plan Workflow

1. `/research mode=debate` → debate agents run
2. Synthesis → `queues/wave_NN_plan.md` created
3. Human reviews and approves the plan
4. `/research mode=queue` → queue file written to `queues/active.txt`
5. `/deploy queues/active.txt`
6. Results collected → `/collect`
7. `/compare` → update KNOWLEDGE.md
8. Loop

## Budget Tracking
Before running any Stage 2 or Stage 3, estimate cost:
- 500 steps: ~$0.5
- 4000 steps: ~$4
- 13780 steps: ~$14

Use `/cost` to check remaining budget before Stage 3.

## Scripts
- `scripts/research.sh` — main orchestrator with all modes
- `scripts/wave_plan.sh` — generates wave_NN_plan.md from debate synthesis
- `scripts/next_wave_number.sh` — finds the next wave number

## Key Files Read
- `KNOWLEDGE.md` — current proven facts and falsified hypotheses
- `research/PIPELINE.md` — the research pipeline definition
- `queues/active.txt` — current active queue (if any)
- `queues/waveNN_plan.md` — most recent wave plan
- `debate/CONCLUSIONS.md` — debate history
