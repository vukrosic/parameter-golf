# 5 Research Threads — One Per L40S Cluster

Target: beat 1.2244 BPB by ≥0.005 (≤1.2194) in 600s on 8xH100.
Each thread is independent. Each gets 1x L40S. Run in parallel.

AttnRes is dead — all variants lose to baseline by step 500 and the gap is widening.

---

## Thread 1: Optimization (HIGHEST CERTAINTY)

**Why**: LR=0.06 already beats default 0.04 by 0.009 BPB at 200 steps. Extending and combining with schedule tuning is the safest path.

**Experiments** (all at 2000 steps, `MAX_WALLCLOCK_SECONDS=0`):

```
# Phase A: Extend top LRs
b1_lr04_2000          MATRIX_LR=0.04                    # control
b1_lr06_2000          MATRIX_LR=0.06                    # current best
b1_lr08_2000          MATRIX_LR=0.08                    # runner up

# Phase B: With best matrix LR, sweep scalar + embed
b2_slr008             MATRIX_LR=<best> SCALAR_LR=0.08
b2_slr016             MATRIX_LR=<best> SCALAR_LR=0.16
b2_elr01              MATRIX_LR=<best> TIED_EMBED_LR=0.1
b2_elr02              MATRIX_LR=<best> TIED_EMBED_LR=0.2

# Phase C: Schedule tuning with best LR triple
b3_wd2400             WARMDOWN_ITERS=2400
b3_wd4000             WARMDOWN_ITERS=4000
b3_mom098             MUON_MOMENTUM=0.98
b3_wu50               WARMUP_STEPS=50
```

**Runtime**: ~4 hrs sequential

---

## Thread 2: Architecture Search (HIGHEST POTENTIAL)

**Why**: Default 9x512 was never optimized. Depth vs width at fixed param count is critical.

**Experiments** (all at 2000 steps):

```
# Phase A: Depth/width sweep (~17M params each, must check <16MB)
a1_9x512              NUM_LAYERS=9  MODEL_DIM=512   # control
a1_12x448             NUM_LAYERS=12 MODEL_DIM=448   # deeper
a1_7x576              NUM_LAYERS=7  MODEL_DIM=576   # wider
a1_15x384             NUM_LAYERS=15 MODEL_DIM=384   # very deep
a1_6x640              NUM_LAYERS=6  MODEL_DIM=640   # very wide

# Phase B: Head config with best depth/width
a2_heads_16_8         NUM_HEADS=16 NUM_KV_HEADS=8
a2_heads_8_8          NUM_HEADS=8  NUM_KV_HEADS=8   # full MHA
a2_heads_8_2          NUM_HEADS=8  NUM_KV_HEADS=2   # aggressive GQA

# Phase C: MLP width
a3_mlp3               MLP_MULT=3
a3_mlp1               MLP_MULT=1
```

**Runtime**: ~5 hrs

---

## Thread 3: Weight Sharing / Depth Recurrence (HIGHEST CEILING)

**Why**: Parameter-limited but compute-allowed. Weight sharing trades compute for depth.

**Requires code changes** before running.

**Experiments** (after implementation):
```
c1_5x2_512            NUM_UNIQUE_LAYERS=5  NUM_CYCLES=2 MODEL_DIM=512
c1_9x1_512            (control = baseline)
c2_5x2_640            NUM_UNIQUE_LAYERS=5  NUM_CYCLES=2 MODEL_DIM=640
c2_4x3_576            NUM_UNIQUE_LAYERS=4  NUM_CYCLES=3 MODEL_DIM=576
```

**Runtime**: ~1 day (implementation + experiments)

---

## Thread 4: QAT (TARGETED)

**Why**: Quant gap at 500 steps is 0.006 BPB. Unknown at 13,780 steps — could be larger.

**Requires code changes** (STE fake-quant in CastedLinear.forward).

**Experiments**:
```
q0_baseline_5000      ITERATIONS=5000   # measure quant gap at scale
q0_baseline_10000     ITERATIONS=10000  # near-full horizon
q1_qat_last30         QAT_START_FRAC=0.7
q1_qat_last50         QAT_START_FRAC=0.5
q1_qat_always         QAT_START_FRAC=0.0
```

**Runtime**: ~6 hrs

---

## Thread 5: Training Tricks Grab Bag (LOW-RISK)

**Why**: Zero-param knobs that stack.

**Experiments** (all at 2000 steps):
```
t1_clip10             GRAD_CLIP_NORM=1.0
t1_clip05             GRAD_CLIP_NORM=0.5
t2_softcap20          LOGIT_SOFTCAP=20
t2_softcap50          LOGIT_SOFTCAP=50
t3_qkgain10          QK_GAIN_INIT=1.0
t3_qkgain20          QK_GAIN_INIT=2.0
t4_seq2048            TRAIN_SEQ_LEN=2048
t6_batch1M            TRAIN_BATCH_TOKENS=1048576
```

**Runtime**: ~5 hrs

---

## Integration

After all threads: take best from each, combine, run 13,780 steps, verify <16MB, submit.

## Common Setup (every cluster)

```bash
cd /workspace && git clone <repo> && cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024
mkdir -p logs
```

**Always set `MAX_WALLCLOCK_SECONDS=0`** for L40S experiments.
