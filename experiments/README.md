# Experiment Specs

`experiments/specs/` contains repo-backed experiment definitions that can be:

- reviewed and edited in git
- indexed by `auto-research`
- used as experiment templates when creating snapshot runs
- linked to research docs in `research/`

Each spec is a JSON manifest with:

- `spec_id`: stable slug used across repo and DB
- `name`: human-readable label
- `spec_type`: `exploration`, `hypothesis`, `finding`, or `benchmark`
- `template`: research template / runner family
- `stage`: `explore`, `validate`, or `full`
- `lane`: optional sub-stage label such as `micro_explore`
- `runner_profile`: optional named runner preset such as `nano_3L128` or `micro_5L192`
- `baseline_spec_id`: optional same-lane baseline reference for delta calculations
- `steps`: requested step count
- `config_overrides`: env-style overrides for the runner
- `promotion_policy`: optional downstream promotion rule, e.g. `top_k_plus_margin`
- `promotion_max`: optional max number of promotions from this spec family
- `promotion_margin_bpb`: optional close-call threshold for promotions
- `linked_docs`: related exploration / hypothesis / finding docs
- `tags`: lightweight grouping labels
- `notes`: intent or execution notes

These files describe experiment intent and config only.

In the main autoresearch workflow:

- snapshot records are the source of truth for runtime state
- dispatch order comes from pending snapshots ordered by `created_at`
- queue status, GPU assignment, and result adjudication do not live in this repo

## Micro-Explore Lane

The repo supports a `micro_explore` lane for reduced-model architecture screening.
This does not replace the normal `explore` stage. Instead:

- `stage` stays `explore`
- `lane` is set to `micro_explore`
- `runner_profile` identifies the fixed reduced model family
- `config_overrides` still contains the exact env vars needed to run the experiment

Current standard runner profiles:

- `nano_3L128`: `NUM_LAYERS=3 MODEL_DIM=128 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=2 MLP_ACT=leaky_relu2_05 ITERATIONS=500 WARMDOWN_ITERS=44 MAX_WALLCLOCK_SECONDS=1200`
- `micro_5L192`: `NUM_LAYERS=5 MODEL_DIM=192 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=2 MLP_ACT=leaky_relu2_05 ITERATIONS=500 WARMDOWN_ITERS=44 MAX_WALLCLOCK_SECONDS=1800`
