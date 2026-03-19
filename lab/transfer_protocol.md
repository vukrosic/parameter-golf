# Transfer Protocol: 1xL40S -> 8xH100

## Why Transfer Works

The training script hardcodes: `grad_accum_steps = 8 // world_size`

| | 1xL40S | 8xH100 |
|---|--------|--------|
| world_size | 1 | 8 |
| grad_accum_steps | 8 | 1 |
| tokens/step | 8 * 65,536 = 524,288 | 8 * 65,536 = 524,288 |
| optimizer update | identical | identical |
| loss at step N | X | X (same) |

The gradient is the same. The optimizer update is the same. The model state at step N
is the same (within bf16 noise). **Per-step loss curves are identical across hardware.**

## What to Verify Before Submitting

1. **Artifact size**: Run the full int8+zlib quantization path. Verify `< 16,000,000 bytes`.
   This can be checked on L40S — quantization is deterministic.

2. **Step count projection**: At ~43ms/step on 8xH100, 600s gives ~13,780 steps.
   If your config changes param count or model shape significantly, the step time may shift.
   Default architecture: 43ms/step. Wider models or more layers may be slower.

3. **Warmdown timing**: The warmdown is time-based (uses elapsed_ms / max_wallclock_ms).
   Set `MAX_WALLCLOCK_SECONDS=600` on H100 — the schedule adapts automatically.

4. **Muon all-reduce**: On 1 GPU, Muon's all-reduce is a no-op. On 8 GPUs, it sums
   updates across ranks. This is mathematically identical (each rank handles 1/8 of params).

## Submission Checklist

```
[ ] Best config identified and tested at 5000+ steps on L40S
[ ] Projected val_bpb at step 13,780 is <= 1.2194
[ ] Artifact size verified under 16MB cap
[ ] train_gpt.py or fork is under 1500 lines
[ ] No env vars required that aren't in the Hyperparameters class
```

## Running on 8xH100

```bash
# Set your winning env vars
export MATRIX_LR=...
export SCALAR_LR=...
# ... other tuned params ...

export MAX_WALLCLOCK_SECONDS=600
export VAL_LOSS_EVERY=1000
export TRAIN_LOG_EVERY=200
export RUN_ID=submission_v1

torchrun --nproc_per_node=8 train_gpt.py
```

## Packaging the Submission

After the run completes:

```bash
# The script produces these automatically:
# - final_model.int8.ptz (compressed model)
# - logs/<RUN_ID>.txt (training log)

# Verify size
ls -la final_model.int8.ptz
wc -c train_gpt.py

# Create submission.json
python3 -c "
import json, os
code_bytes = os.path.getsize('train_gpt.py')
model_bytes = os.path.getsize('final_model.int8.ptz')
print(json.dumps({
    'author': 'YOUR_NAME',
    'github_id': 'YOUR_GITHUB',
    'name': 'YOUR_SUBMISSION_NAME',
    'blurb': 'Description of your approach',
    'date': '$(date -u +%Y-%m-%dT%H:%M:%SZ)',
    'val_loss': FILL_IN,
    'val_bpb': FILL_IN,
    'bytes_total': model_bytes + code_bytes,
    'bytes_code': code_bytes,
}, indent=2))
"
```

## Risk Factors

1. **torch.compile warmup**: On 8xH100, the compile warmup is a fixed cost against a
   shorter total wall-clock. The `warmup_steps` parameter pre-compiles the graph.
   Default is 20 warmup steps. This is ~0.9s of the 600s budget — negligible.

2. **NCCL overhead**: The distributed all-reduce adds latency per step. For this small
   model, it's minimal (~1-2ms/step). Already accounted for in the 43ms/step measurement.

3. **Numerical divergence**: bf16 matmul order may differ between 1-GPU and 8-GPU runs.
   At 13,780 steps, accumulated noise could shift val_bpb by ~0.001-0.003.
   This is within noise — not a risk if your improvement margin is >= 0.005.
