# Local Setup

## Development Hardware

- 1 reference single-GPU benchmark machine (48GB VRAM class)
- all local experiments run single-GPU with automatic grad accumulation
- step dynamics are used as a cheap proxy for the full challenge setting

## Standard Run Entry Point

Use `infra/run_experiment.sh` for normal local runs:

```bash
infra/run_experiment.sh my_test_name 50
infra/run_experiment.sh my_test_name 200
infra/run_experiment.sh my_test_name 13780

MATRIX_LR=0.08 NUM_LAYERS=12 MODEL_DIM=448 \
  infra/run_experiment.sh my_arch_test 200
```

Each run writes:

- `results/<run_id>/summary.json`
- `results/<run_id>/metadata.json`
- `results/<run_id>/train.log`

## Tiered Screening

For broad architecture questions, screen multiple candidates locally before doing longer runs:

```bash
cp screens/template.py screens/<topic>.py
python3 infra/tiered_screen.py --screen screens/<topic>.py --ladder quick
```

The report lands in `results/tiered_screen_<topic>_<date>.md`.

## Git Hook Guardrail

Enable the pre-commit size guard once per clone:

```bash
infra/install_git_hooks.sh
```

Override the file-size cap if needed:

```bash
MAX_FILE_MB=50 git commit -m "..."
```

## Duration Guide

Reference single-GPU timings:

| Steps | Wall-clock | Use for |
|-------|-----------|---------|
| 15 | ~50s | smoke test only |
| 50 | ~2.8 min | quick sanity |
| 200 | ~11 min | short architecture check |
| 500 | ~28 min | medium convergence signal |
| 1000 | ~55 min | stronger comparison |
| 3000 | ~2.8 hr | late-stage check |
| 5000 | ~4.6 hr | near-final validation |
| 13780 | ~12.7 hr | full pre-submission run |

## Interpreting Results

`val_bpb` is the primary competition metric.

- `val_loss` is still useful for debugging
- `train_loss` tracks optimization, not competition standing
- `final_int8_zlib_roundtrip` is the submission-style score to watch on serious runs
