# GPU Step Timing Benchmarks

All measurements use the default 9-layer / 512-dim / 8-head / 4-KV-head setup with batch 524,288 tokens. On single-GPU runs this means `grad_accum=8`; the L40S range spans multiple activation experiments on that same base config.

## Measured Step Times

| GPU | VRAM | Step avg (ms) | 500 steps | 13,780 steps | Source | Date |
|-----|------|--------------|-----------|-------------|--------|------|
| 8xH100 (8 GPUs) | 80GB×8 | ~43 | ~22s | 600s (10 min) | lab/transfer_protocol.md | — |
| 1xL40S | 48GB | ~2,294-2,487 | ~19-21 min | ~8.8-9.5 hrs | monitor_status.log (multiple runs) | 2026-03-19 |
| 1xRTX 3090 (local) | 24GB | ~2,629 | ~22 min | ~10.1 hrs | local smoke + activation runs | 2026-03-21 |
| 1xRTX 3090 (remote, Novita) | 24GB | ~2,702 | ~23 min | ~10.3 hrs | remote activation runs | 2026-03-21 |

## Relative Speed (vs L40S baseline ~2,400ms)

| GPU | Relative speed | Notes |
|-----|---------------|-------|
| L40S | 1.00x | Lab reference GPU |
| RTX 3090 (local) | 0.91x (~9% slower) | |
| RTX 3090 (remote) | 0.89x (~11% slower) | Slightly slower than local 3090, likely thermal/clock |

## Memory Usage

| GPU | Peak allocated | Peak reserved |
|-----|---------------|--------------|
| RTX 3090 (24GB) | 13,830 MiB | 14,150 MiB |

## Pricing & Cost-Efficiency Analysis

### Rental Prices (Novita AI)

All hourly prices below are Novita AI rental rates.

| GPU | $/hr |
|-----|------|
| RTX 3090 (spot) | $0.10 |
| RTX 4090 | $0.17 |
| RTX 3090 (on-demand) | $0.21 |
| L40S | $0.28 |
| RTX 5090 | $0.30 |
| 1xH100 | $1.29 |
| 8xH100 | $10.32 (8 × $1.29) |

### Cost Per Training Run

Step times marked with * are **theoretical estimates** (bandwidth-scaled from L40S). We need to measure real speed when we use these GPUs.

| GPU | Step time (ms) | Hours for 13,780 steps | Cost (full run) | Cost (500 steps) | $/step |
|-----|---------------|----------------------|----------------|-----------------|--------|
| RTX 3090 (spot) | 2,629 | 10.06 hr | **$1.01** | $0.037 | $0.0000730 |
| RTX 5090 | ~1,157* | ~4.43 hr | **$1.33*** | $0.048* | $0.0000964* |
| RTX 4090 | ~2,057* | ~7.87 hr | **$1.34*** | $0.049* | $0.0000971* |
| 1xH100 | ~344* | ~1.32 hr | **$1.70*** | $0.062* | $0.0001232* |
| 8xH100 | 43 | 0.165 hr | **$1.70** | $0.062 | $0.0001232 |
| RTX 3090 (on-demand) | 2,629 | 10.06 hr | **$2.11** | $0.077 | $0.0001534 |
| L40S | 2,400 | 9.19 hr | **$2.57** | $0.093 | $0.0001867 |

### Cost-Efficiency Ranking (cost per step, lower = better)

1. **RTX 3090 (spot $0.10)** — $0.073/1k steps — cheapest by far
2. **RTX 5090* ($0.30)** — $0.096/1k steps — fastest single-GPU, 32% pricier than 3090 spot
3. **RTX 4090* ($0.17)** — $0.097/1k steps — nearly tied with 5090, slower but cheaper hourly
4. **1xH100* ($1.29)** — $0.123/1k steps — same efficiency as 8x, just slower (1.3 hr vs 10 min)
5. **8xH100 ($10.32)** — $0.123/1k steps — same $/step as 1xH100, fastest wall-clock
6. **RTX 3090 (on-demand $0.21)** — $0.153/1k steps — 2.1x worse than spot, worse than H100
7. **L40S ($0.28)** — $0.187/1k steps — worst value overall

### Theoretical Step Time Estimates (4090, 5090, 1xH100)

This model appears **memory-bandwidth-bound** (L40S has 2.5x more compute than 3090 but is only 1.1x faster). Single-GPU estimates use bandwidth scaling from L40S (both support torch.compile):

- **L40S baseline**: 864 GB/s → ~2,400 ms/step
- **RTX 4090**: 1,008 GB/s → ~2,057 ms/step (×0.86 of L40S)
- **RTX 5090**: 1,792 GB/s → ~1,157 ms/step (×0.48 of L40S)
- **1xH100**: estimated ~344 ms/step (8× the 8xH100 step time, sequential grad accum instead of parallel)

**NOTE: These are theoretical. Actual step times could differ significantly due to kernel efficiency, torch.compile behavior, clock speeds, and thermal throttling. We need to measure real speed when we use these GPUs.**

## FLOPS Comparison (RTX 5090 vs L40S)

When training LLMs, BF16 and FP8 are the most relevant precisions. The L40S leverages its enterprise architecture to achieve significantly higher theoretical compute, particularly when utilizing structural sparsity.

| GPU | Architecture | BF16 (Dense / Sparse) | FP8 (Dense / Sparse) | Notes |
|-----|-------------|-----------------------|----------------------|-------|
| **L40S** | Ada Lovelace (Enterprise) | 362 / 733 TFLOPS | 733 / 1,466 TFLOPS | Supports Transformer Engine (FP8/FP16 switching) |
| **RTX 5090** | Blackwell (Consumer) | ~209 TFLOPS (dense)* | ~419 TFLOPS (dense)* | *Tensor Cores w/ accumulate. No sparse acceleration. |

*Note: While the L40S has substantially more raw compute (FLOPS), typical LLM training is often memory-bandwidth-bound. The RTX 5090 has more memory bandwidth (1,792 GB/s) compared to the L40S (864 GB/s), which may make the 5090 faster in practice despite lower theoretical FLOPS.*

## Notes

- L40S range (2,294-2,487ms) reflects variance across different experiments/activations. Some activation functions are slightly faster/slower.
- This file uses the newer March 19-21 measurements; older lab docs that still cite ~3.33s/step on L40S are stale.
- 3090 has sufficient VRAM (uses ~14GB of 24GB available).
- Per-step dynamics (loss curves, gradients) are identical across GPUs — only wall-clock differs.
- The 3090 remote is a Novita cloud instance (proxy.us-ca-6.gpu-instance.novita.ai:62248).
- RTX 3090 runs in eager mode (torch.compile disabled, SM 8.6). All other GPUs support torch.compile.
