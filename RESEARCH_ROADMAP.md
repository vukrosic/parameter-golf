# Parameter Golf: 40-Day Research Roadmap

## Challenge Summary

**OpenAI Parameter Golf**: Train the best language model that fits in a **16MB artifact** (int8+zlib) and trains in **<10 minutes on 8×H100**. Scored by `val_bpb` (bits per byte) on FineWeb validation set.

| Fact | Value |
|------|-------|
| Current record | **1.2244 BPB** (Naive Baseline) |
| Target | **≤1.2194 BPB** (beat by ≥0.005 nats) |
| Theoretical floor | 1.2074 BPB (4hr unlimited, same arch) |
| Artifact limit | ≤16,000,000 bytes (int8 quantized + zlib) |
| Training limit | ≤600 seconds wallclock on 8×H100 |
| Challenge dates | March 18 — April 30, 2026 |

### Hardware

| Machine | Role | Step time (baseline) |
|---------|------|---------------------|
| **1×L40S 48GB** | Development, proxy experiments, scaling law study | ~1815ms/step (524K batch) |
| **8×H100** | Final submission runs only | ~43.5ms/step (524K batch) |

**Critical constraint**: L40S is ~42× slower per step than 8×H100. We develop and validate on L40S, submit on H100. All proxy experiments must be designed so relative rankings transfer across hardware.

---

## Research Philosophy

Adapted from `optimization/ai_research_roadmap.md`:

1. **One knob at a time** — never change multiple variables simultaneously
2. **Cheapest proxy first** — start at 5s, only scale up to confirm findings
3. **Scaling law validation** — a result only counts if it holds across at least 2 compute scales
4. **Decision logging** — every completed sweep gets a `decisions.md` entry with evidence
5. **Kill bad ideas early** — 5-second experiments eliminate 60% of configs before investing more time
6. **Track everything** — append to `experiments.jsonl` for reproducibility

### Source of Truth

- `parameter-golf/experiments.jsonl` — append-only run ledger (create if not exists)
- `parameter-golf/decisions.md` — one decision note per completed sweep
- `parameter-golf/RESEARCH_ROADMAP.md` — this file (update as findings arrive)

---

## Scaling Law Framework

### The Core Idea

**Same model size, vary only training compute.** The model must fit in 16MB int8+zlib regardless — so model architecture is fixed within each experiment. We only change how long we train, and we observe which improvements persist across training durations.

### Compute Ladder (L40S, batch=65536, ~263ms/step)

| Level | Duration | Steps | Tokens | Purpose |
|-------|----------|-------|--------|---------|
| **T1** | **5s** | ~19 | ~1.2M | Ultra-fast screening. Eliminates clearly bad configs. |
| **T2** | **10s** | ~38 | ~2.5M | Confirms T1 rankings. Most architectural effects visible. |
| **T3** | **20s** | ~76 | ~5.0M | Validates scaling. Rankings here are high-confidence. |
| **T4** | **1min** | ~228 | ~15M | LR/schedule effects become visible. |
| **T5** | **5min** | ~1140 | ~75M | Recipe tuning (QAT, warmdown). Near-proxy for H100. |
| **T6** | **15min** | ~3420 | ~224M | Full validation. Closest L40S proxy to H100 10min run. |

**Key assumption to validate**: Rankings at T1-T3 predict rankings at T5-T6 for architecture changes. This is Phase 1's primary question.

### What scales vs what doesn't (hypothesis, to be validated)

| Change type | Expected to scale from T1? | Validate at |
|-------------|---------------------------|-------------|
| Model width (dim) | Yes — more capacity helps immediately | T1→T3 |
| Effective depth (recurrence) | Yes — depth helps from step 1 | T1→T3 |
| Activation (SwiGLU vs ReLU²) | Yes — better gradient flow | T1→T3 |
| KV head count | Yes — attention pattern quality | T1→T3 |
| MLP multiplier | Yes — param efficiency | T1→T3 |
| Learning rate | Partially — needs >100 steps | T3→T5 |
| LR schedule (cosine vs warmdown) | No — needs full training curve | T5→T6 |
| QAT | No — only matters near convergence | T5→T6 |
| Warmdown fraction | No — tail of training only | T5→T6 |

---

## Phase 0: Establish Baselines & Validate Scaling Framework (Days 1-2)

### Question
Do architecture rankings at 5-second proxy predict rankings at 20-second proxy?

### Fixed setup
- **Model size**: Must produce int8+zlib < 16MB (same as H100 target)
- **Batch**: 65536 tokens (fast iteration, 263ms/step on L40S)
- **Warmup**: 3 steps (minimal for T1)
- **Val**: End-of-run only (VAL_LOSS_EVERY=0)
- **Seeds**: 1337 (single seed for screening, add seed 42 for confirmation)

### Experiments

**Step 0.1**: Measure step times at each compute level for baseline config

```
Config: NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2
Run at: T1 (5s), T2 (10s), T3 (20s)
Record: val_bpb, step_ms, params, int8_zlib_size
```

**Step 0.2**: Run 5 diverse architectures at T1, T2, T3 to test ranking stability

| Config | Unique | Effective | Dim | Params (est) | Rationale |
|--------|--------|-----------|-----|-------------|-----------|
| baseline | 9 (none) | 9 | 512 | 17M | Reference |
| recur_small | 3 | 9 | 512 | 6M | Same depth, fewer params |
| recur_deep | 4 | 16 | 640 | 12.5M | More depth via recurrence |
| swiglu_deep | 4 | 16 | 640 | 12.5M | + SwiGLU activation |
| wide | 4 | 16 | 768 | ~18M | Fill 16MB budget |

Run each at T1, T2, T3 = 15 experiments × ~15s avg = **~4 minutes total**.

### Acceptance criteria
- Rank correlation (Spearman) between T1 and T3 rankings is ≥0.8
- If <0.8, T1 is unreliable → increase minimum proxy to T2 or T3
- Document the ranking transfer result in `decisions.md`

### Expected time: ~30 minutes including analysis

---

## Phase 1: Architecture Search via Scaling Ladder (Days 2-5)

### Question
What is the optimal architecture (depth, width, recurrence ratio, activation, KV heads) for the 16MB parameter budget?

### Method
Use the cheapest validated proxy level from Phase 0 (expected: T1 or T2).

### Step 1.1: Recurrence Ratio Sweep (one knob)
Fix: MODEL_DIM=640, NUM_HEADS=10, NUM_KV_HEADS=5, MLP_MULT=2, no SwiGLU

| Name | Unique | Effective | Depth ratio |
|------|--------|-----------|-------------|
| r_2x4 | 2 | 8 | 4× |
| r_2x8 | 2 | 16 | 8× |
| r_3x3 | 3 | 9 | 3× |
| r_3x6 | 3 | 18 | 6× |
| r_4x4 | 4 | 16 | 4× |
| r_4x6 | 4 | 24 | 6× |
| r_5x3 | 5 | 15 | 3× |
| r_5x4 | 5 | 20 | 4× |
| r_6x3 | 6 | 18 | 3× |
| r_none | 9 (none) | 9 | 1× |

Run at: proxy level (T1 or T2) → 10 experiments × ~10s = **~2 minutes**
Confirm top 3 at T3 → 3 × 20s = **1 minute**

**Decision**: Lock optimal recurrence ratio (unique × cycles).

### Step 1.2: Model Width Sweep (one knob)
Fix: Locked recurrence ratio, no SwiGLU

| Dim | Heads | KV | Params (approx) | int8+zlib est |
|-----|-------|----|-----------------|---------------|
| 512 | 8 | 4 | ~6-12M | ~4-8MB |
| 576 | 9 | 3 | ~8-14M | ~5-9MB |
| 640 | 10 | 5 | ~10-16M | ~7-11MB |
| 704 | 11 | 1 | ~11-17M | ~7-12MB |
| 768 | 12 | 4 | ~14-20M | ~9-14MB |
| 832 | 13 | 1 | ~16-22M | ~11-15MB |

Run at proxy → confirm top 3 at T3.
**Reject any config where int8+zlib > 16MB.**

**Decision**: Lock model width.

### Step 1.3: Activation Function (one knob)
Fix: Locked recurrence + width

| Activation | Notes |
|-----------|-------|
| ReLU² | Baseline (current) |
| SwiGLU | ~Parameter-neutral replacement |

Run at proxy → confirm at T3.
Already have evidence SwiGLU wins by ~0.05 BPB at 500 steps.

**Decision**: Lock activation function.

### Step 1.4: KV Head Count (one knob)
Fix: Locked recurrence + width + activation

| KV heads | Style |
|----------|-------|
| 1 | Multi-Query Attention |
| num_heads//2 | Grouped Query (baseline) |
| num_heads | Full Multi-Head |

Run at proxy → confirm at T3.

**Decision**: Lock KV head configuration.

### Step 1.5: MLP Multiplier (one knob)
Fix: Everything else locked

| MLP mult | Notes |
|----------|-------|
| 2 | Baseline |
| 3 | Wider MLP, fewer layers fit in budget |
| 4 | Very wide MLP |

**Decision**: Lock MLP width. Architecture is now fully specified.

### Step 1.6: Scaling Law Confirmation
Run the locked architecture at T1→T2→T3→T4→T5 and plot the loss curve.
Compare with baseline at same compute levels.
The gap should be consistent or growing.

### Expected time: ~2-3 hours total for Phase 1

---

## Phase 2: Hyperparameter Tuning (Days 5-10)

### Question
What is the optimal training recipe for the locked architecture?

### Why after architecture
- LR interacts with model depth/width — wrong to tune LR before locking arch
- Schedule effects only matter at longer training — need T4+ experiments
- QAT only matters near convergence — need T5+ experiments

### Step 2.1: Learning Rate Sweep
Use T4 (1min, ~228 steps) as proxy — LR effects need ≥100 steps.

| LR set | tied_embed_lr | matrix_lr | scalar_lr | Notes |
|--------|---------------|-----------|-----------|-------|
| 0.5× | 0.025 | 0.02 | 0.02 | Conservative |
| 1.0× | 0.05 | 0.04 | 0.04 | Baseline |
| 1.5× | 0.075 | 0.06 | 0.06 | Aggressive |
| 2.0× | 0.10 | 0.08 | 0.08 | Very aggressive |

Run at T4, confirm winner at T5.

**Decision**: Lock LR.

### Step 2.2: LR Schedule
Use T5 (5min) — schedule effects need full training curve.

| Schedule | Notes |
|----------|-------|
| warmdown | Baseline (constant + linear decay at end) |
| cosine | Standard cosine decay |

**Decision**: Lock schedule.

### Step 2.3: Warmdown Fraction
If warmdown schedule wins:

| Fraction | Notes |
|----------|-------|
| 8.7% | Baseline (1200/13780 on H100) |
| 15% | Moderate |
| 20% | Aggressive |

**Decision**: Lock warmdown.

### Step 2.4: Warmup Steps
| Steps | Notes |
|-------|-------|
| 10 | Minimal |
| 20 | Baseline |
| 50 | Extended |

**Decision**: Lock warmup.

### Expected time: ~4-6 hours total for Phase 2

---

## Phase 3: Quantization-Aware Training (Days 10-14)

### Question
When should QAT start, and how much does it close the quant gap?

### Background
- Baseline: 0.007 BPB quant gap (pre-quant 1.2172 → post-quant 1.2244)
- 4hr run: 0.033 BPB quant gap (1.1749 → 1.2074) — gap grows with training
- QAT should close this gap by training with fake int8 quantization

### Method
Use T5 (5min) and T6 (15min) — QAT only matters near convergence.

### Step 3.1: QAT Start Fraction Sweep
| Start frac | When QAT enables | Notes |
|-----------|-------------------|-------|
| 0.0 (off) | Never | Baseline, measures quant gap |
| 0.2 | 20% through training | Very early |
| 0.3 | 30% | Early |
| 0.5 | 50% | Half-way |
| 0.7 | 70% | Late |

Run at T5, confirm at T6.

### Step 3.2: Measure actual quant gap
For the winning QAT config, measure:
- Pre-quant val_bpb
- Post-quant val_bpb (int8+zlib roundtrip)
- Gap should be <0.005 BPB

**Decision**: Lock QAT schedule.

### Expected time: ~3-4 hours for Phase 3

---

## Phase 4: Integration & L40S Validation (Days 14-18)

### Question
Does the full stack (arch + recipe + QAT) deliver the expected improvement at scale?

### Step 4.1: Full L40S Run
Run the complete locked config at T6 (15min, ~3400 steps, ~224M tokens).
This is our best L40S proxy for the H100 10min run.

Compare with baseline at same compute:
- Baseline T6: Expected ~1.35-1.40 BPB (extrapolated)
- Our config T6: Expected ~1.30-1.35 BPB

### Step 4.2: Extended L40S Run (optional)
If time permits, run at 30min and 1hr to see if improvements hold at scale.

### Step 4.3: Multi-GPU Compatibility Check
Verify `train_gpt_golf.py` works with `torchrun --nproc_per_node=8` syntax.
Test on L40S with `--nproc_per_node=1` to verify DDP code path.

### Step 4.4: Size Budget Verification
Confirm int8+zlib < 16,000,000 bytes for the locked config.
If over budget: reduce model width or recurrence depth.

### Expected time: ~2-4 hours for Phase 4

---

## Phase 5: 8×H100 Submission (Days 18-22)

### Step 5.1: Calibration Run
First H100 run: measure actual step time for our architecture.

```bash
MAX_WALLCLOCK_SECONDS=600 TRAIN_BATCH_TOKENS=524288 \
[LOCKED CONFIG] \
torchrun --standalone --nproc_per_node=8 train_gpt_golf.py
```

Record: actual steps completed, step_avg, final val_bpb.

### Step 5.2: Adjust for H100 Throughput
Based on actual step time:
- Calculate actual ITERATIONS for 10min
- Adjust WARMUP_STEPS proportionally
- Adjust warmdown fraction
- Adjust QAT_START_FRAC for actual training length

### Step 5.3: LR Scaling for Multi-GPU
With 8 GPUs, grad_accum goes from 8→1 (same global batch).
LR should NOT change (same effective batch size).
But validate with a quick run.

### Step 5.4: Submission Attempts (2-3 runs)
Run with final config. Verify:
- [ ] Trains in <600 seconds
- [ ] int8+zlib < 16,000,000 bytes
- [ ] val_bpb < 1.2194

### Expected time: ~2-3 hours of H100 time (5-8 runs × 10-20 min each)

---

## Phase 6: Advanced Techniques (Days 22-35)

Only if Phase 5 doesn't beat the record, or to push further.

### 6.1: Novel Architecture Ideas
- **Mixture of Experts** within 16MB budget (fewer active params, same total)
- **Low-rank attention** (reduce KV projection size)
- **Residual scaling** per layer
- **Dynamic depth** (early exit for easy tokens)

### 6.2: Training Tricks
- **Gradient clipping** tuning
- **Beta1/Beta2** tuning for Muon optimizer
- **Sequence length** (train@512 → eval@1024 for more steps per token budget)
- **Data ordering** / curriculum learning

### 6.3: Compression Optimization
- **Structured pruning** to reduce int8+zlib size
- **Mixed precision** quantization (some layers int4, others int8)
- **Custom compression** scheme (not just zlib)

### 6.4: Eval-time Tricks (if allowed)
- **Eval at seq_len=2048** (FAQ says this is allowed)
- **Test-time training** (mentioned in challenge description as valid approach)

### Expected time: Days of experimentation per idea

---

## Phase 7: Paper Writing & Portfolio (Days 35-40)

### 7.1: Document Findings
- Scaling law analysis: what transfers from 5s to 10min
- Architecture insights: optimal recurrence ratio, why SwiGLU helps
- Quantization gap analysis: how QAT closes the gap

### 7.2: Prepare Submission
- Clean `train_gpt_golf.py` to <1500 lines
- Package for records/ directory
- Write submission README

---

## Timeline Summary

| Days | Phase | Key Question | Experiments | Time on L40S |
|------|-------|-------------|-------------|-------------|
| 1-2 | **0: Baselines** | Do 5s rankings predict 20s rankings? | ~15 | ~30 min |
| 2-5 | **1: Architecture** | Optimal depth/width/activation/KV? | ~40 | ~3 hr |
| 5-10 | **2: Hyperparams** | Optimal LR/schedule/warmup? | ~20 | ~5 hr |
| 10-14 | **3: QAT** | When to start QAT? How much gap does it close? | ~10 | ~4 hr |
| 14-18 | **4: Integration** | Does full stack work at scale? | ~5 | ~3 hr |
| 18-22 | **5: H100 Submission** | Beat 1.2244? | ~8 | ~3 hr H100 |
| 22-35 | **6: Advanced** | Push further with novel techniques | open | open |
| 35-40 | **7: Paper** | Document and package | — | — |

**Critical path to first submission attempt: ~15 hours of L40S time + ~3 hours of H100 time = achievable in Days 1-22.**

---

## Experiment Tracking Template

Every experiment should be logged:

```json
{
  "exp_id": "p1s1_r01_recur_4x4",
  "phase": "1.1",
  "question": "What is the optimal recurrence ratio at 640d?",
  "config": {
    "num_unique_blocks": 4,
    "num_effective_layers": 16,
    "model_dim": 640,
    "num_heads": 10,
    "num_kv_heads": 5,
    "mlp_mult": 2,
    "use_swiglu": false
  },
  "compute_level": "T2",
  "batch_tokens": 65536,
  "iterations": 38,
  "seed": 1337,
  "val_bpb": null,
  "int8_bpb": null,
  "step_ms": null,
  "params": null,
  "int8_zlib_bytes": null,
  "status": "ready",
  "notes": ""
}
```

---

## Quick Commands

### Run a 5-second experiment (T1)
```bash
CUDA_VISIBLE_DEVICES=0 TRAIN_BATCH_TOKENS=65536 ITERATIONS=19 \
WARMUP_STEPS=3 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=5 \
[CONFIG VARS] RUN_ID=<name> \
python3 train_gpt_golf.py
```

### Run a 10-second experiment (T2)
```bash
CUDA_VISIBLE_DEVICES=0 TRAIN_BATCH_TOKENS=65536 ITERATIONS=38 \
WARMUP_STEPS=5 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=10 \
[CONFIG VARS] RUN_ID=<name> \
python3 train_gpt_golf.py
```

### Run a 20-second experiment (T3)
```bash
CUDA_VISIBLE_DEVICES=0 TRAIN_BATCH_TOKENS=65536 ITERATIONS=76 \
WARMUP_STEPS=8 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=20 \
[CONFIG VARS] RUN_ID=<name> \
python3 train_gpt_golf.py
```

### Run a 1-minute experiment (T4)
```bash
CUDA_VISIBLE_DEVICES=0 TRAIN_BATCH_TOKENS=65536 ITERATIONS=228 \
WARMUP_STEPS=15 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=50 \
[CONFIG VARS] RUN_ID=<name> \
python3 train_gpt_golf.py
```

---

## Existing Evidence (to be validated, not assumed)

From 500-step experiments (old batch size 524K, ~15min each):

| Finding | Evidence strength | Next step |
|---------|------------------|-----------|
| SwiGLU beats ReLU² by ~0.05 BPB | Medium (500 steps only, 1 seed) | Validate at T1→T3 |
| 4×4=16eff recurrence beats baseline | Medium (500 steps, 1 seed) | Validate at T1→T3 |
| QAT neutral at 500 steps | Low (may help at convergence) | Test at T5→T6 |
| Cosine LR hurts at 500 steps | Low (may help at longer training) | Test at T5→T6 |
| 12.5M params → 7.3MB (room to grow) | High (measured) | Test wider models |

**None of these findings are locked.** They are hypotheses to be validated through the scaling ladder.

---

## Decision Log (append here)

### Day 1: [DATE] — Scaling Law Validation
- `question`: ...
- `evidence`: ...
- `decision`: ...
- `next_step`: ...
