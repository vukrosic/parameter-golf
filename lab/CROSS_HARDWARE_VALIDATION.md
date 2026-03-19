# Cross-Hardware Validation Plan: 1xL40S ↔ 8xH100

## Motivation

All development happens on 1xL40S. Submission runs on 8xH100 with a 600s wall-clock cap.
We claim per-step dynamics are identical. **We must verify this empirically before submitting.**

If the claim is wrong, we waste the H100 run. If it's right, we can develop confidently on L40S.

---

## What Transfers (Theory)

| Property | Transfers? | Why |
|----------|-----------|-----|
| Batch size | Yes | `grad_accum_steps = 8 // world_size` → always 524,288 tokens/step |
| Optimizer state | Yes | Same grads, same updates, same state |
| Loss at step N | Yes* | *Within bf16 accumulation noise |
| Model weights at step N | Yes* | *Same caveat — matmul order differs |
| Warmdown schedule | **NO** | Time-based: `elapsed_ms / max_wallclock_ms`. Must use `MAX_WALLCLOCK_SECONDS=0` + `ITERATIONS=N` on L40S |
| torch.compile graphs | Maybe | Compiler may choose different kernels per GPU arch |
| Step timing | No | ~2.1s/step (L40S) vs ~43ms/step (H100) |
| Total steps in 600s | No | ~285 (L40S) vs ~13,780 (H100) |

## What Could Go Wrong

### 1. Numerical divergence over long horizons
- bf16 matmul on L40S (Ada Lovelace) vs H100 (Hopper) may accumulate differently
- 13,780 steps of accumulated rounding → potential drift
- **Risk level**: LOW. Expected divergence ~0.001-0.003 BPB based on transfer_protocol.md estimates
- **Mitigation**: measure empirically at matched step counts

### 2. torch.compile producing different kernels
- Different GPU architectures → different kernel selections → different numerical behavior
- Could affect gradient flow through fused operations
- **Risk level**: LOW-MEDIUM. Functionally equivalent but numerically different
- **Mitigation**: compare loss curves at matched steps

### 3. NCCL all-reduce ordering
- On 1 GPU: Muon all-reduce is a no-op
- On 8 GPU: ring-reduce sums across ranks in a specific order
- For this small model: negligible effect
- **Risk level**: VERY LOW
- **Mitigation**: included in the baseline comparison

### 4. Memory pressure differences
- L40S: 46GB VRAM, single process
- H100: 80GB per GPU × 8, but DDP overhead
- Some modes (weighted, weighted_vector) store all layer outputs → more VRAM
- **Risk level**: LOW for default arch. Check if experimental modes fit in H100 memory
- **Mitigation**: monitor peak VRAM on L40S, ensure < 70GB total for 8-GPU case

### 5. Data loading bottleneck
- 8xH100 needs 8× the data throughput at same step rate
- If data loading is CPU-bound on the H100 machine, step time increases
- **Risk level**: LOW. Dataset is small (fineweb-edu subset), likely fits in RAM
- **Mitigation**: check step timing stability in first 100 steps of H100 baseline

---

## Research Questions

### Q1: Does val_bpb at step N match between L40S and H100?
- **Hypothesis**: Within 0.003 BPB at all matched checkpoints
- **Method**: Run identical config for 500 steps on both, compare val_bpb every 50 steps
- **Success**: max |bpb_l40s - bpb_h100| < 0.005 at any checkpoint
- **Failure mode**: systematic offset or diverging curves

### Q2: Does the ranking of configs transfer?
- **Hypothesis**: If config A beats config B by X BPB on L40S at step N, it also beats B on H100 at step N by approximately X BPB
- **Method**: Run baseline + best variant on both, compare delta
- **Success**: ranking is preserved; delta magnitude within 50% of L40S measurement
- **This is the critical question** — even if absolute values differ, ranking must transfer

### Q3: Does the warmdown schedule behave correctly under time-based control?
- **Hypothesis**: `MAX_WALLCLOCK_SECONDS=600` on H100 triggers warmdown at the right step
- **Method**: Check logs for warmdown start step on H100; compare to `ITERATIONS - WARMDOWN_ITERS`
- **Success**: warmdown starts within 50 steps of expected
- **Note**: On L40S we always use `MAX_WALLCLOCK_SECONDS=0` + explicit `ITERATIONS`, so warmdown is step-based. On H100 it's time-based. This is the biggest transfer risk.

### Q4: Does int8+zlib artifact size match?
- **Hypothesis**: Artifact size is deterministic given model weights; hardware doesn't matter
- **Method**: Compare artifact sizes from L40S vs H100 at matched steps
- **Success**: byte-for-byte identical artifacts

### Q5: Do AttnRes variants show the same relative behavior on H100?
- **Hypothesis**: If value_residual_gated beats baseline by 0.01 BPB at step 2000 on L40S, it shows similar improvement on H100
- **Method**: Run winning AttnRes variant + baseline on H100 for 2000 steps
- **This validates our entire research program**

---

## Experiment Protocol

### Transferability Rules (maximize comparability)

1. **Always use `MAX_WALLCLOCK_SECONDS=0` on L40S** — forces step-based scheduling, removes time-based warmdown ambiguity
2. **Always use explicit `ITERATIONS=N`** — never rely on wall-clock to determine training length
3. **Fix the random seed** — default seed should be identical; verify with `grep -r seed train_gpt.py`
4. **Log val_bpb at matched step intervals** — use `VAL_LOSS_EVERY=50` for short runs, `VAL_LOSS_EVERY=500` for long runs
5. **Same CHECKPOINT_EVERY** — enable resumability and matched comparison points
6. **Same env vars** — document every env var used in each run
7. **No time-dependent hyperparameters on L40S** — warmdown is the only time-dependent param; always override with step-based control
8. **Record ms/step** — needed to project H100 wall-clock for a given step count

### Phase H1: L40S Baseline (run NOW, while AttnRes experiments proceed)

Already done: `baseline_500` → 1.4540 BPB (int8) at 500 steps.

Still needed:
```bash
# 2000-step baseline on L40S (validate longer curve)
RUN_ID=h1_baseline_2000 ITERATIONS=2000 VAL_LOSS_EVERY=50 TRAIN_LOG_EVERY=100 \
  CHECKPOINT_EVERY=500 MAX_WALLCLOCK_SECONDS=0 python train_gpt.py
```

### Phase H2: H100 Baseline (when H100 access available)

```bash
# Short validation run: 500 steps with step-based control (not 600s wall-clock)
RUN_ID=h2_baseline_500 ITERATIONS=500 VAL_LOSS_EVERY=50 TRAIN_LOG_EVERY=100 \
  CHECKPOINT_EVERY=500 MAX_WALLCLOCK_SECONDS=0 \
  torchrun --nproc_per_node=8 train_gpt.py

# Then: full 600s run with wall-clock control
RUN_ID=h2_baseline_600s MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=500 \
  TRAIN_LOG_EVERY=200 \
  torchrun --nproc_per_node=8 train_gpt.py
```

### Phase H3: Cross-Compare

After H1 and H2 complete:
1. Extract val_bpb at steps 50, 100, 150, 200, 250, 300, 350, 400, 450, 500
2. Compute max absolute difference
3. Compute rank correlation (should be 1.0)
4. If max diff > 0.005 at any checkpoint, investigate before trusting further L40S results

### Phase H4: Config Transfer Validation

Run winning config from AttnRes experiments on H100:
```bash
# Step-matched run (same step count as L40S winner validation)
RUN_ID=h4_winner_matched ATTNRES_MODE=<winner> ITERATIONS=<same_as_l40s> \
  VAL_LOSS_EVERY=50 MAX_WALLCLOCK_SECONDS=0 \
  torchrun --nproc_per_node=8 train_gpt.py
```

Compare val_bpb curves. If ranking transfers, proceed to full submission run.

### Phase H5: Full Submission Run

```bash
RUN_ID=submission_v1 ATTNRES_MODE=<winner> MATRIX_LR=<best> \
  MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=1000 TRAIN_LOG_EVERY=200 \
  torchrun --nproc_per_node=8 train_gpt.py
```

Verify:
- [ ] val_bpb <= 1.2194 (int8+zlib roundtrip)
- [ ] artifact size < 16,000,000 bytes
- [ ] training completed within 600s
- [ ] warmdown triggered at correct point

---

## Analysis Script Template

```bash
# Compare L40S vs H100 at matched steps
# Usage: bash compare_hardware.sh logs/h1_baseline_2000.txt logs/h2_baseline_500.txt

L40S_LOG=$1
H100_LOG=$2

echo "=== L40S ==="
grep "val_bpb" "$L40S_LOG" | head -10

echo ""
echo "=== H100 ==="
grep "val_bpb" "$H100_LOG" | head -10

echo ""
echo "=== Side-by-side (first 500 steps) ==="
paste <(grep "val_bpb" "$L40S_LOG" | head -10 | awk '{print $2, $4}') \
      <(grep "val_bpb" "$H100_LOG" | head -10 | awk '{print $4}') | \
  column -t
```

---

## Decision Matrix

| Outcome | Action |
|---------|--------|
| Q1 pass, Q2 pass | Full confidence in L40S development. Proceed to submission. |
| Q1 fail, Q2 pass | Rankings transfer despite absolute offset. Adjust target by measured offset. |
| Q1 pass, Q2 fail | Absolute values match but rankings differ. Something architecture-specific. Investigate. |
| Q1 fail, Q2 fail | Cannot trust L40S results. Must develop directly on H100. |

---

## Timeline

This validation should happen **before** committing to a full 13,780-step submission run.
Minimum viable validation: Phase H1 (L40S, 30 min) + Phase H2 (H100, ~25s for 500 steps) + Phase H3 (analysis, 5 min).

Total H100 time needed for validation: ~30 seconds (500-step run) + 600 seconds (full baseline).
This is cheap — do it before any tuned submission run.
