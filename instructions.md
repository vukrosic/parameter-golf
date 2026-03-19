# 5-second GPU run

```bash
cd /root/llm-research-kit/parameter-golf
env CUDA_VISIBLE_DEVICES=0 NO_COMPILE=1 SKIP_EVAL=1 MAX_WALLCLOCK_SECONDS=5 TRAIN_BATCH_TOKENS=65536 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=1 WARMUP_STEPS=1 ITERATIONS=100000 RUN_ID=quick_5s python3 train_gpt_golf.py 2>&1 | tee results/quick_5s/train.log
python3 plot_log.py results/quick_5s/train.log results/quick_5s/quick_5s.png
```
