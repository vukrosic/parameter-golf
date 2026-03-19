# Research Roadmap

**Goal**: Beat 1.2244 BPB by >= 0.005 (target: <= 1.2194 BPB) in 600s on 8xH100.

**Methodology**: All development on 1xL40S. Per-step dynamics transfer exactly to 8xH100.
We compare configs by val_bpb at matched step counts, not wall-clock time.

**Current baseline reference**: 1.2244 BPB at step 13,780 (default config, no tuning).

---

## Phase 0: Calibration (Day 1)

**Purpose**: Establish ground truth loss curves to compare everything against.

### Experiments

- [ ] `p0_baseline_500`: Default config, 500 steps. Establishes the per-step loss curve.
- [ ] `p0_baseline_1000`: Default config, 1000 steps. Confirm curve shape.

### Deliverable
A baseline loss curve with val_bpb measured every 50 steps for the first 1000 steps.
All future experiments compare against this curve.

---

## Phase 1: Learning Rate Tuning (Days 1-2)

**Purpose**: Find optimal LR for the ~13,780 step training horizon.

**Key constraint**: The previous attempt failed by tuning LR on 4-8 steps.
We must tune at a representative horizon. 500 steps (3.6% of full run) is the minimum
for reliable signal. 1000 steps is better.

### Experiments

**Matrix LR sweep** (Muon optimizer — this is the dominant LR):
- [ ] `p1_mlr_0.02`: MATRIX_LR=0.02 (0.5x default)
- [ ] `p1_mlr_0.04`: MATRIX_LR=0.04 (1x default — same as baseline)
- [ ] `p1_mlr_0.06`: MATRIX_LR=0.06 (1.5x)
- [ ] `p1_mlr_0.08`: MATRIX_LR=0.08 (2x)
- [ ] `p1_mlr_0.12`: MATRIX_LR=0.12 (3x)
- [ ] `p1_mlr_0.16`: MATRIX_LR=0.16 (4x)

Run each for 500 steps. Pick top 3, extend to 1000 steps.

**Scalar LR sweep** (with best matrix LR):
- [ ] `p1_slr_0.02`: SCALAR_LR=0.02
- [ ] `p1_slr_0.04`: SCALAR_LR=0.04 (default)
- [ ] `p1_slr_0.08`: SCALAR_LR=0.08
- [ ] `p1_slr_0.16`: SCALAR_LR=0.16

**Embedding LR sweep** (with best matrix + scalar LR):
- [ ] `p1_elr_0.02`: TIED_EMBED_LR=0.02
- [ ] `p1_elr_0.05`: TIED_EMBED_LR=0.05 (default)
- [ ] `p1_elr_0.1`: TIED_EMBED_LR=0.1
- [ ] `p1_elr_0.2`: TIED_EMBED_LR=0.2

### Deliverable
Optimal LR triple (matrix, scalar, embed) validated at 1000 steps.

### How to know we're done
The best config beats `p0_baseline_500` by a clear margin at step 500.
Ranking is stable between step 500 and step 1000 measurements.

---

## Phase 2: Schedule Tuning (Days 2-3)

**Purpose**: Optimize warmup and warmdown for the 13,780-step horizon.

### Experiments

**Warmdown sweep** (with best LRs from Phase 1):
- [ ] `p2_wd_600`: WARMDOWN_ITERS=600
- [ ] `p2_wd_1200`: WARMDOWN_ITERS=1200 (default)
- [ ] `p2_wd_2400`: WARMDOWN_ITERS=2400
- [ ] `p2_wd_4000`: WARMDOWN_ITERS=4000

Note: Warmdown is time-based in the code. With MAX_WALLCLOCK_SECONDS=600 on 8xH100,
warmdown_iters=1200 means warmdown starts when remaining_ms <= 1200 * step_ms.
At ~43ms/step, that's ~51.6s before the end (~12,580 of 13,780 steps).

These experiments need 3000+ steps to see warmdown effects.

**Warmup sweep**:
- [ ] `p2_wu_0`: WARMUP_STEPS=0
- [ ] `p2_wu_20`: WARMUP_STEPS=20 (default)
- [ ] `p2_wu_50`: WARMUP_STEPS=50

**Muon momentum**:
- [ ] `p2_mom_0.90`: MUON_MOMENTUM=0.90
- [ ] `p2_mom_0.95`: MUON_MOMENTUM=0.95 (default)
- [ ] `p2_mom_0.98`: MUON_MOMENTUM=0.98

### Deliverable
Optimal schedule parameters. Run best config for 3000-5000 steps.

---

## Phase 2.5: AttnRes — Inter-layer Attention (PRIORITY)

**Purpose**: Upgrade cross-layer information routing. The baseline already has `resid_mix` (embed only)
and U-Net `skip_weights` (encoder→decoder). AttnRes generalizes this to ALL previous layers.

**Key insight**: The model already has primitive cross-layer connections. AttnRes is the principled upgrade.

### Experiments (run with best LR from Phase 1)

- [ ] `attnres_cumsum`: Cumulative attention output residual (0 extra params), 500 steps
- [ ] `attnres_value_residual`: Cross-layer value residual from layer 0, 500 steps
- [ ] `attnres_weighted`: Softmax-weighted combination of all prev layer outputs (45 params), 500 steps
- [ ] `attnres_weighted_vector`: Per-dimension weights (23K params), 500 steps
- [ ] Winner extended to 1000+ steps
- [ ] Winner + 12-layer architecture (deeper model enabled by better routing)

### Details
See `/lab/experiments/attnres/PLAN.md` (full implementation strategy) and `/lab/research_papers/attnres.md`

### Deliverable
Best AttnRes variant with val_bpb improvement quantified. Sponsored content post with before/after results.

---

## Phase 3: Architecture Search (Days 3-5)

**Purpose**: Find better depth/width/head tradeoffs within the 16MB cap.

### Size-Matched Architectures

All configs must produce artifacts under 16,000,000 bytes after int8+zlib.
The default (9x512, 17M params) uses ~15.86MB.

Candidates (approximate param counts):
- [ ] `p3_7x576`: NUM_LAYERS=7, MODEL_DIM=576 (~17M, wider+shallower)
- [ ] `p3_12x448`: NUM_LAYERS=12, MODEL_DIM=448 (~17M, deeper+narrower)
- [ ] `p3_15x384`: NUM_LAYERS=15, MODEL_DIM=384 (~15M, very deep)
- [ ] `p3_6x640`: NUM_LAYERS=6, MODEL_DIM=640 (~18M, check cap!)

### Head Configuration
- [ ] `p3_heads_16_8`: NUM_HEADS=16, NUM_KV_HEADS=8 (more heads, same GQA ratio)
- [ ] `p3_heads_8_2`: NUM_HEADS=8, NUM_KV_HEADS=2 (aggressive GQA)
- [ ] `p3_heads_8_8`: NUM_HEADS=8, NUM_KV_HEADS=8 (full MHA)

### MLP Width
- [ ] `p3_mlp_3`: MLP_MULT=3 (hidden=1536 vs default 1024)
- [ ] `p3_mlp_1`: MLP_MULT=1 (hidden=512, minimal MLP)

### Deliverable
Best architecture with LR re-tuned. Run for 1000+ steps.

---

## Phase 4: Advanced Techniques (Days 5-7)

**Purpose**: Techniques that may improve training efficiency or reduce quant gap.

### Quantization-Aware Training (QAT)
The 4h run showed 0.0325 BPB quant gap. Reducing this is free performance.
- [ ] `p4_qat_basic`: Add int8 quantization noise during training
- [ ] `p4_qat_anneal`: QAT with noise annealing (less noise as training progresses)

### Gradient Clipping
- [ ] `p4_clip_1.0`: GRAD_CLIP_NORM=1.0
- [ ] `p4_clip_0.5`: GRAD_CLIP_NORM=0.5

### Other Knobs
- [ ] `p4_softcap_20`: LOGIT_SOFTCAP=20
- [ ] `p4_softcap_50`: LOGIT_SOFTCAP=50
- [ ] `p4_qkgain_1.0`: QK_GAIN_INIT=1.0
- [ ] `p4_qkgain_2.0`: QK_GAIN_INIT=2.0

### Deliverable
Best config with all improvements stacked. Run for 3000+ steps.

---

## Phase 5: Final Validation (Days 7-8)

**Purpose**: Full-length runs on 1xL40S to project 8xH100 performance.

- [ ] `p5_final_13780`: Best config, 13,780 steps (~12.7 hours on L40S)
- [ ] `p5_final_quant`: Verify int8+zlib artifact size under 16MB
- [ ] `p5_final_verify`: Verify val_bpb post-quant is <= 1.2194

### Deliverable
A config ready for 8xH100 submission with projected val_bpb.

---

## Phase 6: 8xH100 Submission

- [ ] Run winning config on 8xH100 with MAX_WALLCLOCK_SECONDS=600
- [ ] Verify artifact size, val_bpb
- [ ] Package submission (train_gpt.py + final_model.int8.ptz + submission.json)

---

## Research Log

Record observations, surprises, and pivots here as work progresses.

| Date | Phase | Observation |
|------|-------|-------------|
| 2026-03-19 | P1 | MATRIX_LR=0.06 leading (+0.006 BPB over baseline at 200 steps) |
| 2026-03-19 | P2.5 | Paper read. Baseline already has resid_mix + skip_weights — AttnRes upgrades these |
| 2026-03-19 | P2.5 | 4 variants designed: cumsum (0 params) → value_res → weighted (45) → weighted_vec (23K) |
| 2026-03-19 | P2.5 | Phase 1 AttnRes complete: cumsum hurts (-0.007 BPB at 200 steps), value_residual consistently worse but gap closing (0.037→0.004 over 500 steps). Crossover projected ~step 1500-2000. |
| 2026-03-19 | P2.5 | PLAN_v2 written: 5 phases, gated VR, mid-layer VR, weighted variants, scale to 13,780 steps. ~16h GPU total. |
| 2026-03-19 | Post | "When Better Architecture Is Actually Just Different Initialization" — post drafted in lab/posts/ |
