# GPU Step Timing Benchmarks

All measurements on default config: 9 layers, 512 dim, 8 heads, 4 KV heads, batch 524,288 tokens, grad_accum=8 on single GPU.

## Measured Step Times

| GPU | VRAM | Step avg (ms) | 500 steps | 13,780 steps | Source | Date |
|-----|------|--------------|-----------|-------------|--------|------|
| 8xH100 (8 GPUs) | 80GB×8 | ~43 | ~22s | 600s (10 min) | transfer_protocol.md | — |
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

## Pricing

| GPU | Spot ($/hr) | On-demand ($/hr) | Cost per 500 steps | Cost per 13k steps |
|-----|------------|-----------------|-------------------|-------------------|
| RTX 3090 (Novita) | $0.10 | $0.21 | ~$0.04 spot | ~$1.03 spot |

## Notes

- L40S range (2,294-2,487ms) reflects variance across different experiments/activations. Some activation functions are slightly faster/slower.
- 3090 has sufficient VRAM (uses ~14GB of 24GB available).
- Per-step dynamics (loss curves, gradients) are identical across GPUs — only wall-clock differs.
- The 3090 remote is a Novita cloud instance (proxy.us-ca-6.gpu-instance.novita.ai:62248).
