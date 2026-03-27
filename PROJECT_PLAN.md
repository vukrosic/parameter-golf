# Parameter Golf — Project Plan

## Goal

Win the OpenAI Parameter Golf Challenge: best 16MB language model, trained in under 10 minutes on 8xH100s.

**Metric**: `val_bpb` (lower is better)
**Current best**: See `KNOWLEDGE.md` for latest frontier
**Leaderboard baseline**: 1.2244 BPB

## Research Methodology

See `autoresearch/` for how to run structured research in this repo:
- `autoresearch/README.md` — system overview and repo map
- `autoresearch/RESEARCH_LOOP.md` — mechanical steps of a research cycle
- `autoresearch/PROMPTS.md` — ready-to-use prompts for agent sessions

## Current Status

See `KNOWLEDGE.md` for proven facts and latest results.

## Working Model

1. Edit code here in `/root/parameter-golf/`
2. Rsync to GPU, run experiments
3. Pull results back to `results/`
4. Analyze, update `KNOWLEDGE.md`, design next batch
