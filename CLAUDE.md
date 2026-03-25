# Parameter Golf Repo Notes

## What This Is

A competitive ML research repo for the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf): train the best 16MB language model in under 10 minutes on 8xH100s, scored by `val_bpb` (lower is better).

This repo now holds:

- competition training code
- local experiment helpers
- research notes, findings, and screen configs
- archived leaderboard submissions in `records/`

The old repo-local queue, fleet, debate, and GPU orchestration workflow has been retired. This repo is no longer the place to manage dispatch state.

## Quick Setup

```bash
git clone <repo_url> && cd parameter-golf
pip install -r requirements.txt
python3 data/cached_challenge_fineweb.py --variant sp1024
```

## Project Structure

```text
train_gpt.py                 # main competition training script
train_gpt_mlx.py             # Apple Silicon variant
KNOWLEDGE.md                 # persistent research memory
data/                        # dataset/tokenizer helpers
infra/run_experiment.sh      # canonical single-run entry point
infra/tiered_screen.py       # cheap multi-candidate local screen
research/                    # ideas, explorations, hypotheses, findings
screens/                     # tiered-screen configs
records/                     # leaderboard submissions
results/                     # local experiment outputs
```

## Running Experiments

Start with a local screen whenever the question is broad:

```bash
cp screens/template.py screens/<topic>.py
python3 infra/tiered_screen.py --screen screens/<topic>.py --ladder quick
```

Run a single configured experiment like this:

```bash
infra/run_experiment.sh <name> <steps>
MATRIX_LR=0.08 NUM_LAYERS=12 MODEL_DIM=448 infra/run_experiment.sh my_arch_test 200
```

Key outputs:

- `results/<run_id>/summary.json`
- `results/<run_id>/metadata.json`
- `results/<run_id>/train.log`

## Conventions

- `val_bpb` is the only competition metric that matters
- read `KNOWLEDGE.md` before proposing new work
- `infra/FORBIDDEN.md` defines the research constraints
- experiment names use snake_case
- local manual orchestration has been removed from this repo; keep this repo focused on code and evidence
