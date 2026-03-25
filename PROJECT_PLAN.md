# Parameter Golf Repo Scope

Parameter Golf is now the code-and-evidence repo for the challenge work. The old repo-local queue, fleet, and debate workflow has been retired from this directory.

What stays here:

- training code (`train_gpt.py`, `train_gpt_mlx.py`)
- local run helpers (`infra/run_experiment.sh`, `infra/tiered_screen.py`)
- experiment specs in `experiments/specs/`
- research notes and findings in `research/`
- leaderboard records in `records/`

What no longer lives here:

- repo-local execution queues
- repo-local GPU fleet control
- repo-local debate and wave automation

This repo should be treated as:

1. the source tree for model code and configs
2. the place where research claims and evidence are written down
3. a dependency that external orchestration can read or copy

## Working Model

- use `infra/run_experiment.sh` for direct local runs
- use `infra/tiered_screen.py` for cheap local multi-candidate screens
- record conclusions in `KNOWLEDGE.md` and `research/`
- keep orchestration state outside this repo

## Directory Map

```text
train_gpt.py
infra/run_experiment.sh
infra/tiered_screen.py
experiments/specs/
research/
records/
results/
```

## Rule

If a file in this repo assumes local queues, local dispatch, or repo-owned fleet state, it is stale and should be removed or rewritten.
