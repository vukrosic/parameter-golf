---
name: post
description: Draft X/Twitter posts from experiment results or debate outputs. Generates formatted posts with metrics, comparisons, and images.
---

# X/Twitter Post Generation

## Usage

### `/post` (no args)
Draft a post from the most recent completed experiment or wave.

### `/post stage=1`
Post Stage 1 (Explore) results. Format:
```
🚨 Parameter Golf Experiment Results

Quick screen (500 steps, n=X ideas)

🎯 Best: <name> — <val_bpb> (Δ<delta> BPB vs baseline)
💀 Dropped: <list>

📊 Next: <winning idea> → 2000-step validation
```

### `/post stage=2`
Post Stage 2 (Validate) results. Includes learning curve description if available.
```
📈 Parameter Golf — Validation Results

<idea> at <steps> steps (n=2 seeds)
Seed 1337: <val_bpb>
Seed 42:   <val_bpb>
Mean:      <val_bpb>

vs best known: <compare>
✅ Promising — scaling to full run
❌ Dead end — <reason>
```

### `/post stage=3`
Post Stage 3 (Full run) results.
```
🏁 Parameter Golf — Full Run Complete

Config: <model description>
Pre-quant:  <bpb>
Post-quant: <bpb>

Leaderboard baseline: 1.2244
Our result:           <bpb> (Δ<delta>)

Status: <submitted / pending>
```

### `/post from=<experiment>`
Draft from a specific experiment's results.

## Include Images
If `research/figures/<name>.png` exists, include it in the post.

## Scripts
- `scripts/post.sh` — main post drafting script

## Output
Writes drafted post to stdout. User reviews and edits before posting manually.
