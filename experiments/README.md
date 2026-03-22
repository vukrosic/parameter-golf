# Experiment Specs

`experiments/specs/` contains repo-backed experiment definitions that can be:

- reviewed and edited in git
- indexed by `auto-research`
- queued from the web UI
- linked to research docs in `research/`

Each spec is a JSON manifest with:

- `spec_id`: stable slug used across repo and DB
- `name`: human-readable label
- `spec_type`: `exploration`, `hypothesis`, `finding`, or `benchmark`
- `template`: research template / runner family
- `stage`: `explore`, `validate`, or `full`
- `steps`: requested step count
- `config_overrides`: env-style overrides for the runner
- `linked_docs`: related exploration / hypothesis / finding docs
- `tags`: lightweight grouping labels
- `notes`: intent or execution notes
- `desired_state`: `draft`, `queued`, `paused`, or `archived`

`auto-research` treats these files as the source of truth for experiment design.
Runtime state stays in the platform database.
