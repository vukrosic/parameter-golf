---
name: setup
description: Set up parameter-golf on a fresh machine — install dependencies, download data, configure git, and run a smoke test. Works for both local and remote GPU machines.
---

# Setup Parameter Golf

Automate the full setup process for a new machine.

## Instructions

### Local setup (default, no args)

Run the setup script:
```bash
bash .claude/skills/setup/scripts/setup.sh
```

Add `--smoke-test` to also run a 5-step training test:
```bash
bash .claude/skills/setup/scripts/setup.sh --smoke-test
```

Report what was installed and whether the smoke test passed.

### Remote GPU setup (e.g., `/setup remote ARCH5`)

1. Get the GPU's port and password from `bash .claude/skills/fleet/scripts/discover_gpus.sh`
2. Run the remote setup script:
   ```bash
   bash .claude/skills/setup/scripts/setup_remote.sh <port> <password>
   ```
3. Report the GPU specs and confirm training works.

### What it does
1. Installs Python deps from `requirements.txt`
2. Downloads FineWeb sp1024 dataset (skips if already present)
3. Installs git hooks for file size guards
4. Optional smoke test (5 training steps)

## Key Scripts
- `scripts/setup.sh` — Local machine setup
- `scripts/setup_remote.sh` — Remote GPU setup via SSH
