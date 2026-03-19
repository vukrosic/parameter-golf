# Architecture Follow-Up Plan — 2026-03-19

## What the current research says

This is not random search.

- The head sweep already established one durable gain: `NUM_HEADS=8 NUM_KV_HEADS=8`
  at `MATRIX_LR=0.06` beat the `8/4` control at both `400` and `1000` steps.
- The old depth/width sweep was also directional: `12x448` slightly beat `9x512`
  by step `1000`, while `15x384` lagged. That points to a moderate move toward
  more depth, not an extreme deep-and-narrow regime.
- The live batch is probing that same region and already shows ordering, with
  `10x448` and `9x480` ahead of the more aggressive deep/narrow trials.

## Why the current batch is not enough by itself

The live architecture runs are using the default `MATRIX_LR=0.04`, while the best
measured attention topology in this repo uses `MATRIX_LR=0.06`.

That means the current five jobs are still useful for ranking architectures, but
they are not a fair best-shot sweep for beating the record.

## Queued experiments

The queue in [queue.txt](/root/parameter-golf/lab/queue.txt) does two things:

1. `ctrl_L9_D512_H8_KV8_M2_lr06_2000`
   Extends the current best-known measured config instead of betting everything on
   unproven variants.
2. Four `1200`-step ridge runs around the likely sweet spot:
   `12x448`, `11x432`, `10x448`, `9x480`, all with `8/8` heads and `MATRIX_LR=0.06`.

## Step budget

- `2000` steps for the direct control extension
- `1200` steps for each architecture sweep run
- Total queued work: `6800` training steps

This keeps enough breadth to find a better architecture while still pushing one
known-good baseline deeper.
