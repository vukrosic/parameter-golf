# When "Better Architecture" Is Actually Just Different Initialization

We tested value residual connections on a 17M-parameter GPT model (9 layers, 512 dim) training on FineWeb-10B, and the results were not what we expected.

## The Setup

Value residuals are a simple architectural tweak: store the value embeddings from layer 0 and add them to every subsequent layer's attention values. The paper claims this reduces attention concentration in deeper layers and speeds training. Zero extra parameters.

Our baseline already has two forms of cross-layer routing:
- `resid_mix`: learnable blend of each layer's input with the embedding
- `skip_weights`: U-Net-style encoder-decoder skip connections

We added value residuals on top and ran both configs for 500 identical steps (524K tokens/step, Muon optimizer, Pre-LN RMSNorm).

## The Result

Value residuals were **worse at every single checkpoint** -- but the gap is closing:

```
Step 100:  baseline 1.9422  |  value_res 1.9795  |  gap: 0.037 BPB
Step 200:  baseline 1.6324  |  value_res 1.6545  |  gap: 0.022 BPB
Step 300:  baseline 1.5384  |  value_res 1.5488  |  gap: 0.010 BPB
Step 500:  baseline 1.4481  |  value_res 1.4522  |  gap: 0.004 BPB
```

The convergence rate of the gap is roughly logarithmic. If it continues, the lines cross somewhere around step 1500-2000.

## Why This Happens

Value residuals inject layer-0's value embedding into every subsequent layer. At step 1, layer 0's values are random noise. By adding random noise to all other layers' values, you're making the *entire model's attention mechanism worse* for the first few hundred steps.

As training progresses, layer 0's values become meaningful representations, and injecting them becomes helpful. The crossover point depends on how quickly layer 0 learns useful value representations vs. how much damage the noise injection does early on.

This is not an initialization problem in the usual sense (weight init). It's a *architectural coupling* problem: the value residual creates a dependency where 8 layers are bottlenecked on 1 layer's quality. Early in training, that bottleneck is pure drag. Late in training, it may become an information superhighway.

## The Confound That Fooled Us

Our first result showed value residuals *winning* at 200 steps. This was a warmdown artifact: with ITERATIONS=200, the learning rate schedule starts decaying early. The value residual variant happened to be in a better position when warmdown hit. When we ran both variants for 500 steps with the same schedule, the truth emerged.

Lesson: always compare at matched step counts with matched schedules, not at matched wallclock or iteration configs that trigger different warmdown behavior.

## What We're Testing Next

The narrowing gap suggests value residuals may help at longer training horizons (our target is 13,780 steps). We're also testing:
- Learned per-layer scalar gates on the value residual (let each layer decide how much layer-0 signal to use)
- Softmax-weighted combination of ALL previous layers' values (45 params)
- Per-dimension routing weights (23K params)
- Value residual from layer N/2 instead of layer 0 (skip the "random noise" phase)

The hypothesis: the current implementation is too blunt. Layer 0 alone isn't the right source. A learned routing mechanism that can attend to all previous layers' values should capture the benefit without the early-training penalty.

## Config for Reproduction

```bash
# Baseline (500 steps, 1xL40S)
ITERATIONS=500 VAL_LOSS_EVERY=50 python train_gpt.py

# Value residual variant
ATTNRES_MODE=value_residual ITERATIONS=500 VAL_LOSS_EVERY=50 python train_gpt.py
```

Model: 9 layers, 512 dim, 8 heads (4 KV), GQA, tied embeddings, 1024 vocab, Muon + Adam, MATRIX_LR=0.04, batch=524K tokens.
