# 5-second GPU run

```bash
cd /root/llm-research-kit/parameter-golf
mkdir -p results/quick_5s
env CUDA_VISIBLE_DEVICES=0 NO_COMPILE=1 SKIP_EVAL=1 MAX_WALLCLOCK_SECONDS=5 \
    TRAIN_BATCH_TOKENS=65536 VAL_LOSS_EVERY=5 VAL_MAX_TOKENS=524288 \
    TRAIN_LOG_EVERY=1 WARMUP_STEPS=1 ITERATIONS=100000 \
    RUN_ID=quick_5s python3 train_gpt_golf.py 2>&1 | tee results/quick_5s/train.log
python3 plot_log.py results/quick_5s/train.log results/quick_5s/quick_5s.png
```

# How to run experiments (for AI models)

All hyperparams are set via env vars. Override any combo in the `env` prefix.

**Key knobs:** `MATRIX_LR` (Muon LR, default 0.04), `EMBED_LR` (0.6), `SCALAR_LR` (0.04), `NUM_LAYERS` (9), `MODEL_DIM` (512), `NUM_HEADS` (8), `NUM_KV_HEADS` (4), `MLP_MULT` (2), `LR_SCHEDULE` (warmdown|cosine), `WARMDOWN_ITERS` (1200), `TRAIN_BATCH_TOKENS` (524288), `TRAIN_SEQ_LEN` (1024).

**Running a sweep:** Loop over values, save each to its own dir, plot:

```bash
cd /root/llm-research-kit/parameter-golf
for lr in 0.01 0.02 0.04 0.08 0.16; do
  mkdir -p results/lr_sweep_${lr}
  env CUDA_VISIBLE_DEVICES=0 NO_COMPILE=1 SKIP_EVAL=1 MAX_WALLCLOCK_SECONDS=5 \
      TRAIN_BATCH_TOKENS=65536 VAL_LOSS_EVERY=5 VAL_MAX_TOKENS=524288 \
      TRAIN_LOG_EVERY=1 WARMUP_STEPS=1 ITERATIONS=100000 \
      MATRIX_LR=$lr RUN_ID=lr_sweep_${lr} \
      python3 train_gpt_golf.py 2>&1 | tee results/lr_sweep_${lr}/train.log
done
```

**Plotting:** `python3 plot_log.py <log_file> <output.png>` plots train+val loss vs step and time. For multi-run comparisons, write a quick matplotlib script parsing all logs (see `results/lr_sweep_comparison.png` for example).

**Scaling up:** Once 5s ranks params, confirm at 10s/20s/40s. Only final val matters — use `VAL_LOSS_EVERY=0` and grep the last `val_loss` line. Rankings have been stable across timescales so far.

```bash
# Final-val-only sweep at longer duration (no intermediate val, faster)
for lr in 0.01 0.02 0.04 0.08 0.16; do
  mkdir -p results/lr_sweep_40s_${lr}
  env CUDA_VISIBLE_DEVICES=0 NO_COMPILE=1 SKIP_EVAL=1 MAX_WALLCLOCK_SECONDS=40 \
      TRAIN_BATCH_TOKENS=65536 VAL_LOSS_EVERY=0 VAL_MAX_TOKENS=524288 \
      TRAIN_LOG_EVERY=1 WARMUP_STEPS=1 ITERATIONS=100000 \
      MATRIX_LR=$lr RUN_ID=lr_sweep_40s_${lr} \
      python3 train_gpt_golf.py 2>&1 | tee results/lr_sweep_40s_${lr}/train.log
done
# Extract final val BPB from each
for lr in 0.01 0.02 0.04 0.08 0.16; do
  echo -n "LR=$lr "; grep "val_bpb" results/lr_sweep_40s_${lr}/train.log | tail -1
done
```

**Tips:**
- Use `VAL_LOSS_EVERY=5` for plots, `VAL_LOSS_EVERY=0` for speed when you only need final val
- 5s runs (~22 steps) are enough to see relative LR/architecture trends
- Rankings stable from 5s→40s so far — screen at 5s, confirm at 40s
- For final scoring, drop `SKIP_EVAL=1` and use full 600s runs
