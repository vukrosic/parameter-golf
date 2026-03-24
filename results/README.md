# Results Tracking Convention

Use one folder per experiment run:

`results/<run_id>/`

Recommended contents:

- `summary.json` (commit): key outcomes (`val_bpb`, quantized score, artifact bytes)
- `metadata.json` (commit): run config, seed, git commit, hardware snapshot, timestamp
- `notes.md` (optional commit): quick interpretation and follow-up actions
- `train.log` (do not commit): full raw training log for local debugging

`lab/run_experiment.sh` now generates `summary.json` and `metadata.json` automatically.

## What to Commit vs Ignore

Commit:

- small structured outputs needed to reproduce conclusions
- markdown analysis and plots under a few MB

Do not commit directly:

- raw checkpoints, `.pt`/`.ptz` payloads, and large log dumps
- temporary TensorBoard/W&B outputs

For large artifacts, keep them in object storage or Git LFS and record links in `notes.md`.
