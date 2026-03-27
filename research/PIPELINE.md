# Parameter Golf Research Method

This file describes the research method only. `parameter-golf` is not the authority for dispatch order or runtime state; the autoresearch lab owns that through snapshot records.

## Core Loop

Use the repo in four stages:

1. **Screen wide**
   Use `infra/tiered_screen.py` or short local runs to eliminate obviously bad ideas.
2. **Explore**
   Run a promising candidate at roughly 500 steps and compare against a relevant baseline.
3. **Validate**
   Re-run real winners at 2000-4000 steps, ideally on multiple seeds.
4. **Full**
   Only spend a full 13780-step run on configurations that have already survived validation.

## Step Counts

| Stage | Steps | Purpose | Typical gate |
|-------|------:|---------|--------------|
| Screen | 1-200 | kill obvious losers cheaply | clear sign only |
| Explore | 500 | quick architecture signal | >0.01 BPB improvement |
| Validate | 2000-4000 | confirm a real effect | >0.005 BPB and reproducible |
| Full | 13780 | final pre-submission check | beats target |

## Principles

- never spend full-run compute on an unscreened idea
- compare against a real baseline, not a memory of one
- write down the claim before treating a result as meaningful
- call out noise, caveats, and size-budget confounds explicitly

## Where To Record Things

- `research/explorations/`: free-form probes and short-result notes
- `research/hypotheses/`: pre-registered claims and test plans
- `research/findings/`: cleaned-up writeups once the evidence is stable
- `KNOWLEDGE.md`: current working memory of wins, failures, and open questions

## Retirement Note

If a document or script in this repo still assumes:

- repo-owned execution queues
- repo-owned GPU fleet state
- debate-generated wave plans
- local `/deploy` or `/collect` orchestration

then it is stale and should be removed or rewritten.
