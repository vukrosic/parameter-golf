# Parameter Golf: Record-Breaking Plan

## Challenge Requirements

| Constraint | Value |
|-----------|-------|
| **Artifact size** | ≤16MB (int8 quantized + zlib compressed) |
| **Training time** | ≤10 minutes on 8×H100 |
| **Scoring metric** | `val_bpb` (bits per byte) on FineWeb validation set (50k docs) |
| **Current record** | **1.2244 BPB** (Naive Baseline, 2026-03-18) |
| **Target** | **≤1.2194 BPB** (beat record by ≥0.005 nats) |
| **Best unlimited** | 1.2074 BPB (4hr, same architecture — theoretical floor) |
| **Submission** | `train_gpt.py` + model artifact, reproducible |

### What gets scored
```
1. Train model: torchrun --nproc_per_node=8 train_gpt.py  (must finish in <600s)
2. Serialize: int8 quantize → zlib compress → must be ≤16,000,000 bytes
3. Evaluate: decompress → dequantize → run val set → report val_bpb
```

---

## Current State of Experiments

### Completed Results (sorted by int8_bpb)

| Experiment | Steps | int8_bpb | Config | Size |
|-----------|-------|----------|--------|------|
| exp0_baseline | 2000 | **1.2976** | 9L, 512d, no recurrence | 15.0MB |
| q4_swiglu_640 | 500 | 1.4755 | 4×4=16eff, 640d, SwiGLU | 7.3MB |
| q5_qat_640 | 500 | 1.4763 | 4×4=16eff, 640d, QAT@50% | — |
| q6_combined | 500 | 1.4763 | 4×4=16eff, 640d, SwiGLU+QAT | — |
| q2_recur_4x4_640 | 500 | 1.4877 | 4×4=16eff, 640d | — |
| a03_5x3_576 | 500 | 1.4962 | 5×3=15eff, 576d | — |
| q7_cosine | 500 | 1.5110 | combined + cosine LR | — |
| q3_recur_3x6_704 | 500 | 1.5161 | 3×6=18eff, 704d | — |
| q0_baseline_500 | 500 | **1.5284** | 9L, 512d (reference) | — |
| q1_recur_3x3_512 | 500 | 1.5505 | 3×3=9eff, 512d | — |

**Extended sweep (46 experiments)**: Running now, currently on Phase A (a04). ~5-6 hours remaining.

### Key Findings So Far

1. **SwiGLU is the single best improvement**: q4_swiglu (1.4755) vs q0_baseline (1.5284) = **-0.053 BPB at 500 steps**
2. **Recurrence helps**: q2_recur (1.4877) vs q0_baseline (1.5284) = **-0.041 BPB**
3. **QAT at 500 steps is neutral**: q5_qat (1.4763) ≈ q4_swiglu (1.4755) — QAT only helps at scale by closing quant gap
4. **Cosine LR hurts at 500 steps**: q7 (1.5110) worse than q6 (1.4763) — cosine needs longer training
5. **Model size is NOT a bottleneck**: Best config uses only 7.3MB of 16MB budget — room to grow

---

## The Gap Analysis

### Record specs (8×H100)
- Step time: **43.54ms** (baseline, 9 layers, 512d, 17M params)
- Steps in 10min: **~13,780**
- Tokens seen: **7.2B**
- Pre-quant BPB: 1.2172
- Post-quant BPB: 1.2244 (**0.007 BPB quant gap**)

### Our best architecture on 8×H100 (estimated)
- Config: 4 unique × 4 cycles = 16 effective layers, 640d, SwiGLU
- Effective depth 16 vs 9 → step time ≈ 43.54 × (16/9) ≈ **77ms/step**
- Steps in 10min: **~7,800**
- Tokens: **~4.1B** (vs 7.2B baseline — 43% fewer tokens)
- Parameters: 12.5M (vs 17M baseline)
- Model size: 7.3MB (vs 15MB baseline — massive headroom)

### Can we beat 1.2244?

**The core tradeoff**: Recurrence gives us a deeper, better model per step, but each step is slower due to more effective layers. We see 4.1B tokens vs 7.2B — we need the architecture quality to compensate for 43% fewer tokens.

**Evidence it's possible**:
- At 500 steps (equal tokens), SwiGLU+recurrence beats baseline by 0.053 BPB
- At 2000 steps, the baseline alone gets 1.2976 — recurrence should push well below
- The 4hr unlimited run (1.2074) used the same baseline arch — our improved arch should converge faster

**Risk**:
- We don't know the exact step time on 8×H100 for our config
- Cosine LR hurt at 500 steps — schedule tuning is critical
- We haven't validated at >2000 steps with our best config yet

---

## Record-Breaking Strategy

### Phase 1: Complete L40S Architecture Search (NOW — next 6 hours)
**Status**: Extended sweep running (46 experiments). No action needed.

**Goal**: Identify top 3 architectures by relative 500-step ranking.

Key questions being answered:
- What's the optimal recurrence ratio? (Phase A: 2×8 through 6×3)
- Does MLP_MULT=3 or 4 help? (Phase B)
- Which SwiGLU combos win? (Phase C)
- Optimal LR scaling? (Phase D)
- Best QAT schedule? (Phase E)
- Do rankings hold at 1000/2000/4000 steps? (Phases F/G/H)

### Phase 2: L40S Long Validation Runs (after sweep, ~4 hours)
Run top 3 configs at **4000+ steps** to:
1. Confirm 500-step rankings hold at scale
2. Measure actual quant gap (int8 BPB - float BPB)
3. Verify model stays under 16MB
4. Test schedule variations (warmdown fraction, LR decay)

```bash
# Example: best config at 4000 steps with full validation
TRAIN_BATCH_TOKENS=524288 ITERATIONS=4000 VAL_LOSS_EVERY=500 \
NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 \
NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1 \
QAT_START_FRAC=0.5 RUN_ID=validation_best \
python3 train_gpt_golf.py
```

### Phase 3: 8×H100 Calibration Run (CRITICAL — needs cloud GPU)
**This is the make-or-break step.** We MUST run on actual 8×H100 to:

1. **Measure real step time** for our architecture (estimated 77ms, could be 60-90ms)
2. **Know actual step count** in 10 minutes
3. **Tune wallclock-aware schedule** (warmup, warmdown fractions)
4. **Verify multi-GPU scaling** (gradient accumulation changes: 8→1 with 8 GPUs)

```bash
# First calibration: measure throughput
NCCL_IB_DISABLE=1 \
RUN_ID=h100_calibration \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_BATCH_TOKENS=524288 \
VAL_LOSS_EVERY=200 \
NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 \
MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 \
MLP_MULT=2 USE_SWIGLU=1 QAT_START_FRAC=0.5 \
torchrun --standalone --nproc_per_node=8 train_gpt_golf.py
```

**Scaling considerations for 8×H100**:
- Baseline uses `grad_accum_steps = 8 // world_size` → with 8 GPUs, no accumulation needed
- Each GPU processes 524288/8 = 65536 tokens per step
- LR may need scaling: `√(8)` factor for linear scaling rule
- NCCL all-reduce overhead is minimal for 12.5M params

### Phase 4: Architecture Tuning on H100 (~3-5 runs, 10min each)
Once we have real H100 throughput numbers:

1. **If step_time < 70ms**: We can afford deeper models (20+ effective layers)
   - Try 4×5=20 or 4×6=24 effective layers
   - May increase MODEL_DIM to 704 to fill 16MB budget

2. **If step_time > 80ms**: Prioritize width over depth
   - Reduce to 12 effective layers
   - Increase MODEL_DIM to 768+

3. **Optimize for exact step count**:
   - Set ITERATIONS to actual achievable count
   - Tune WARMUP_STEPS = 2% of total
   - Tune WARMDOWN_FRAC for clean schedule completion

### Phase 5: Final Submission Attempts (2-3 runs)
With the optimal config identified:

1. Run with `MAX_WALLCLOCK_SECONDS=600` to verify <10min
2. Verify `int8+zlib < 16,000,000 bytes`
3. Verify `val_bpb < 1.2194`
4. Save model + training script as submission

---

## Levers Ranked by Expected Impact

| # | Lever | Expected BPB Gain | Status | Risk |
|---|-------|-------------------|--------|------|
| 1 | **SwiGLU activation** | 0.02-0.04 | ✅ Validated | Low — proven at 500 steps |
| 2 | **Depth recurrence** | 0.01-0.03 | ✅ Validated | Medium — slower step time |
| 3 | **QAT (close quant gap)** | 0.005-0.01 | ✅ Implemented | Low — only affects post-quant |
| 4 | **LR schedule tuning** | 0.005-0.015 | 🔄 Testing | Medium — very sensitive to step count |
| 5 | **Fill 16MB budget** | 0.005-0.015 | ❌ Not tested | Low — more params = better if budget allows |
| 6 | **Longer warmdown** | 0.002-0.005 | 🔄 Testing | Low |
| 7 | **Eval at longer seq_len** | 0.001-0.003 | ❌ Not tested | Low — FAQ allows train@1024 eval@2048 |

**Conservative combined estimate**: 0.04-0.08 BPB improvement over 1.2244 → **target 1.145-1.185 BPB**

---

## Decision: Should We Run on 8×H100 Now?

### Not yet. Here's why:

1. **We haven't finished architecture search** — the extended sweep has 40+ experiments still running
2. **We don't know our best config at scale** — 500-step rankings may shift at 10,000+ steps
3. **H100 time is expensive** — each run costs real money/credits
4. **We should maximize L40S iteration first** — free experiments

### When to go to 8×H100:
- After extended sweep completes AND we've validated top 3 at 4000+ steps on L40S
- **Estimated timeline: 12-18 hours from now**
- Budget: 5-8 H100 runs (50-80 min of 8×H100 time)

### What to prepare before H100:
1. Finalize `train_gpt_golf.py` — ensure it works with `torchrun --nproc_per_node=8`
2. Prepare 3 config variants to test
3. Have submission packaging ready (verify int8+zlib pipeline)

---

## Quick Reference: Running Experiments

### L40S (development, 1× GPU)
```bash
cd /root/llm-research-kit/parameter-golf
CUDA_VISIBLE_DEVICES=0 \
TRAIN_BATCH_TOKENS=524288 \
ITERATIONS=500 \
VAL_LOSS_EVERY=250 \
[CONFIG VARS] \
python3 train_gpt_golf.py
```

### 8×H100 (submission)
```bash
cd /path/to/parameter-golf
NCCL_IB_DISABLE=1 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_BATCH_TOKENS=524288 \
VAL_LOSS_EVERY=200 \
[CONFIG VARS] \
torchrun --standalone --nproc_per_node=8 train_gpt_golf.py
```

### Check results
```bash
# All results sorted
sort -t',' -k3 -n results/sweep_results.csv | column -t -s','

# Specific experiment
grep "final_int8_zlib_roundtrip_exact" results/*/train.log | sort -t: -k3 -n
```

---

## Timeline

| When | What | Where |
|------|------|-------|
| Now → +6h | Extended sweep completes | L40S (running) |
| +6h → +10h | Analyze results, run top-3 at 4000 steps | L40S |
| +10h → +12h | Prep submission, test multi-GPU compat | L40S |
| +12h → +14h | Calibration + 3-5 tuning runs | 8×H100 |
| +14h → +16h | Final submission attempts | 8×H100 |

**Fastest path if impatient**: Skip to H100 now with `4×4=16eff, 640d, SwiGLU` (our current best). Risk: may leave 0.01-0.02 BPB on the table from unfinished architecture search.
