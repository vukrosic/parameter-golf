# Parameter Golf — AI Research Automation

## What This Is

A competitive ML research repo for the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf): train the best 16MB language model in <10 min on 8xH100s, scored by bits-per-byte (val_bpb, lower = better).

The repo doubles as an **automated AI research platform** — Claude Code skills (`/fleet`, `/deploy`, `/status`, etc.) orchestrate a fleet of remote GPUs via SSH for continuous experimentation.

## Quick Setup (New Machine)

```bash
git clone <repo_url> && cd parameter-golf
git checkout lab
pip install -r requirements.txt
python3 data/cached_challenge_fineweb.py --variant sp1024
```

Or use the `/setup` skill which does all of this plus smoke test.

## Project Structure

```
train_gpt.py              # Main training script (CUDA, the competition artifact)
train_gpt_mlx.py          # Apple Silicon variant
requirements.txt           # Python dependencies
data/                      # Dataset download scripts + tokenizers
  cached_challenge_fineweb.py   # Main data downloader
results/                   # Git-tracked experiment results (submission.json, config.json)
records/                   # Official leaderboard submissions
lab/                       # Experiment orchestration
  run_experiment.sh        # Run a single experiment with env var overrides
  run_queue.sh             # Run a queue file of experiments sequentially
  gpu_creds.sh             # SSH credentials for GPU fleet (GITIGNORED)
  gpu_scheduler.py         # Daemon: watches queue.txt, assigns jobs to idle GPUs
  watch_all_gpus.sh        # Fleet monitoring (utilization, temp, cost, progress)
  gpu_sync_cron.sh         # 2-hourly git sync across all GPUs
  export_experiment_artifacts.py  # Exports commit-friendly result files
  experiments/ideas/       # Documented architecture ideas (Tier 1-3)
  gpu_queues/              # Queue files for different GPU targets
utils/                     # Monitoring and plotting helpers
.claude/skills/            # Claude Code skills for GPU orchestration
```

## GPU Fleet Architecture

GPUs are managed via SSH through a proxy host. Credentials live in `lab/gpu_creds.sh` (gitignored) with this pattern:
```bash
HOST="proxy.host.example"
GPU_<NAME>_PORT=<port>
GPU_<NAME>_PASS="<password>"
```

Skills auto-discover GPUs by parsing `gpu_creds.sh` — no hardcoded GPU lists.

### Fleet Operations (Skills)

| Skill | Purpose |
|-------|---------|
| `/setup` | Set up a fresh machine (deps, data, smoke test) |
| `/fleet` | Show all GPU status (utilization, temp, experiment, cost) |
| `/deploy` | Send experiments to GPUs (queue file or single) |
| `/status` | Detailed training progress (step, loss, ETA) |
| `/collect` | Pull results from all GPUs immediately |
| `/compare` | Rank experiments by val_bpb, diff configs |
| `/cost` | Budget report ($40 limit, burn rate, per-GPU) |
| `/kill-exp` | Stop running experiments |
| `/add-gpu` | Register a new GPU instance |

## Running Experiments

### Single experiment
```bash
lab/run_experiment.sh <name> <steps>
# With overrides:
MATRIX_LR=0.08 NUM_LAYERS=12 lab/run_experiment.sh my_test 200
```

### Queue of experiments
```bash
# Queue file format: <name> <steps> [ENV=val ...]
bash lab/run_queue.sh lab/queue_ordered.txt
```

### Key env vars for train_gpt.py
`MATRIX_LR`, `SCALAR_LR`, `EMBED_LR`, `NUM_LAYERS`, `MODEL_DIM`, `NUM_HEADS`, `NUM_KV_HEADS`, `MLP_MULT`, `WARMDOWN_ITERS`, `WARMUP_STEPS`, `LOGIT_SOFTCAP`, `QK_GAIN_INIT`, `ROPE_BASE`, `MUON_MOMENTUM`, `GRAD_CLIP_NORM`, `TIED_EMBED_LR`, `TIED_EMBED_INIT_STD`

## Conventions

- **val_bpb** is the only metric that matters for the competition
- Experiment names use snake_case: `act_leaky05_gradfloor_2000`
- Results go in `results/<name>/` with `submission.json`, `config.json`, `hparams.json`
- Queue files live in `lab/` named `queue_*.txt`
- The `lab` branch is the working branch; `main` is for PRs upstream
- Step counts: 50=smoke, 200=screening, 500-1000=validation, 3000-5000=final, 13780=full H100 equivalent
- Budget: $40 total across all GPUs, tracked in `/cost` skill

## Data

Download with: `python3 data/cached_challenge_fineweb.py --variant sp1024`
- Default: full val split + 80 train shards (8B tokens)
- Smaller: add `--train-shards 1` for local smoke tests
- Data lands in `data/datasets/fineweb10B_sp1024/` and `data/tokenizers/`
