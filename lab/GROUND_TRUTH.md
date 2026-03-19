# Ground Truth

Hard facts derived from actual measurements. Nothing here is estimated or assumed.

## Challenge Rules

- **Track**: 10min / 16MB (beat 1.2244 BPB by >= 0.005 nats)
- **Metric**: `val_bpb` after int8+zlib roundtrip (bits-per-byte, tokenizer-agnostic)
- **Artifact cap**: 16,000,000 bytes (model int8+zlib + code)
- **Time cap**: 600 seconds wallclock on 8xH100
- **Code cap**: train_gpt.py must stay under 1500 lines

## Current Records

| Entry | val_bpb | Steps | Wallclock | Hardware |
|-------|---------|-------|-----------|----------|
| Naive Baseline | 1.2244 | 13,780 | 600s | 8xH100 (pgut1) |
| 4h Unlimited | 1.2074 (post-quant) | 329,430 | 14,400s | 8xH100 (pgut3) |
| 4h pre-quant | 1.1749 | 329,430 | 14,400s | 8xH100 (pgut3) |

**Quant gap**: 1.1749 -> 1.2074 = 0.0325 BPB lost to int8+zlib quantization.

## Hardware Throughput (Measured)

| Hardware | world_size | grad_accum | Batch (tokens) | ms/step | Steps in 600s |
|----------|-----------|------------|-----------------|---------|---------------|
| 8xH100 | 8 | 1 | 524,288 | 43.5 | ~13,780 |
| 1xL40S | 1 | 8 | 524,288 | 3,330 | ~180 |

**Throughput ratio**: 8xH100 is ~76.5x faster per step than 1xL40S.

## Transfer Physics: 1xL40S -> 8xH100

The training script uses `grad_accum_steps = 8 // world_size`:
- 1 GPU: 8 micro-steps of 65,536 tokens = 524,288 tokens/step
- 8 GPU: 1 micro-step of 65,536 tokens/GPU = 524,288 tokens/step

**What is identical per step:**
- Effective batch size (524,288 tokens)
- Model architecture and parameter count
- Optimizer state updates (Muon + Adam)
- Data stream (deterministic with same seed)
- Loss curve (within bf16 numerical noise)

**What differs:**
- Wall-clock per step (3.3s vs 43ms)
- Total steps in 600s (180 vs 13,780)
- Muon all-reduce is trivial on 1 GPU (no-op vs NCCL)
- torch.compile warmup time relative to total budget

**Conclusion**: Any experiment measuring per-step behavior transfers perfectly.
The ONLY thing that doesn't transfer is wall-clock scheduling (warmdown by time).
All schedules must be specified in steps, calibrated for ~13,780 total steps.

## Baseline Architecture (Default Config)

```
vocab_size:     1024  (SP-1024 BPE tokenizer)
num_layers:     9     (4 encoder + 5 decoder, U-Net skip connections)
model_dim:      512
num_heads:      8
num_kv_heads:   4     (GQA, 2:1 ratio)
mlp_mult:       2     (relu^2 MLP, hidden=1024)
tie_embeddings: True
rope_base:      10000
logit_softcap:  30.0
qk_gain_init:   1.5
```

Parameters: 17,059,912 (~17M)
Artifact size: ~15.86 MB (under 16MB cap)

## Optimizer Setup

- **Embedding**: Adam, lr=0.05 (tied_embed_lr)
- **Matrix params** (attention/MLP weights): Muon, lr=0.04
- **Scalar params** (norms, scales, gains): Adam, lr=0.04
- **Warmdown**: Linear decay over last `warmdown_iters` steps (default 1200)
- **Muon momentum**: 0.95 (warmup from 0.85 over 500 steps)

## Size Budget

With default config, ~15.86 MB used of 16 MB cap.
Headroom: ~140 KB for architecture changes.
Any architecture change must be checked against the 16MB cap.

## What Killed the Previous Attempt

LR=30.0 was tuned on 4-8 steps (13-25 seconds), then planned for "~180 steps" on 8xH100.
Actual 8xH100 throughput: ~13,780 steps. The plan was wrong by 76x on step count.
At step 37 with LR=30.0: val_bpb=2.6449 (catastrophically worse than 1.2244 baseline).
The baseline with LR=0.04 runs 13,780 steps and converges properly.
