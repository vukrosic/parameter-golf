"""Single-process tiered screening. Loads CUDA + data once, loops configs."""
import os, sys, time, json, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

# One-time CUDA setup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_mem_efficient_sdp, enable_math_sdp
enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
device = torch.device("cuda", 0)
torch.cuda.set_device(device)

# Import model building blocks (no main() call)
os.environ.setdefault("ITERATIONS", "1")  # prevent any accidental full run
import importlib
import train_gpt as tg

print("Loading data...", flush=True)
SEQ_LEN = 32
BATCH_TOKENS = 256
SEED = 1337
VOCAB_SIZE = 1024
DATA_PATH = "./data/datasets/fineweb10B_sp1024"
TRAIN_FILES = os.path.join(DATA_PATH, "fineweb_train_*.bin")
VAL_FILES = os.path.join(DATA_PATH, "fineweb_val_*.bin")

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

train_loader = tg.DistributedTokenLoader(TRAIN_FILES, 0, 1, device)
print("Data loaded.", flush=True)

GRAD_ACCUM = 8
GRAD_SCALE = 1.0 / GRAD_ACCUM

def run_config(name, steps, **kwargs):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    num_layers      = kwargs.get("num_layers", 1)
    model_dim       = kwargs.get("model_dim", 16)
    num_heads       = kwargs.get("num_heads", 2)
    num_kv_heads    = kwargs.get("num_kv_heads", 1)
    mlp_mult        = kwargs.get("mlp_mult", 1)
    tie_embeddings  = kwargs.get("tie_embeddings", True)
    attnres_mode    = kwargs.get("attnres_mode", "none")
    mlp_act         = kwargs.get("mlp_act", "relu2")
    embed_bottleneck= kwargs.get("embed_bottleneck", 0)
    conv_kernel     = kwargs.get("conv_kernel", 0)
    num_experts     = kwargs.get("num_experts", 0)

    model = tg.GPT(
        vocab_size=VOCAB_SIZE,
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_mult=mlp_mult,
        tie_embeddings=tie_embeddings,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        attnres_mode=attnres_mode,
        mlp_act=mlp_act,
        act_power=2.0,
        act_gate_floor=0.5,
        embed_bottleneck=embed_bottleneck,
        num_unique_blocks=0,
        num_cycles=1,
        conv_kernel=conv_kernel,
        num_experts=num_experts,
        resid_scale_init=1.0,
        stoch_depth_rate=0.0,
        highway_net=False,
        skip_weight_init=1.0,
    ).to(device).bfloat16()
    tg.restore_low_dim_params_to_fp32(model)

    # Simple Adam for everything (no Muon compile overhead)
    opt = torch.optim.Adam(model.parameters(), lr=0.04)

    model.train()
    losses = []
    t0 = time.perf_counter()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(GRAD_ACCUM):
            x, y = train_loader.next_batch(BATCH_TOKENS, SEQ_LEN, GRAD_ACCUM)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            (loss * GRAD_SCALE).backward()
            step_loss += loss.item() * GRAD_SCALE
        opt.step()
        losses.append(step_loss)
    elapsed = time.perf_counter() - t0

    avg_loss = sum(losses) / len(losses)
    os.makedirs(f"results/{name}", exist_ok=True)
    result = {
        "run_id": name, "stage": "tiered_1s",
        "steps": steps, "elapsed_s": round(elapsed, 3),
        "train_loss": round(avg_loss, 4),
        "arch": kwargs
    }
    with open(f"results/{name}/summary.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  {name:30s}  loss={avg_loss:.4f}  ({elapsed:.1f}s)", flush=True)
    return avg_loss

ALL_CONFIGS = {
    "baseline":    {},
    "untied":      {"tie_embeddings": False},
    "moe2":        {"num_experts": 2},
    "moe4":        {"num_experts": 4},
    "attnres_vr":  {"attnres_mode": "value_residual"},
    "attnres_w":   {"attnres_mode": "weighted"},
    "conv3":       {"conv_kernel": 3},
    "swiglu":      {"mlp_act": "swiglu"},
    "heads4":      {"num_heads": 4, "num_kv_heads": 2},
    "deep2":       {"num_layers": 2},
}

ARCH_DESC = {
    "baseline":   "baseline",
    "untied":     "untied embeddings",
    "moe2":       "soft MoE 2 experts",
    "moe4":       "soft MoE 4 experts",
    "attnres_vr": "value residual attn",
    "attnres_w":  "weighted attn residual",
    "conv3":      "depthwise conv k=3",
    "swiglu":     "SwiGLU activation",
    "heads4":     "4 heads / 2 kv-heads",
    "deep2":      "2 layers",
}

# ── STAGE 1: all 10 at 1 step ─────────────────────────────────────────────
print("═" * 60)
print("STAGE 1 — 1s (1 step) — all 10 candidates")
print("═" * 60)
s1_results = {}
for key, kwargs in ALL_CONFIGS.items():
    loss = run_config(f"tiered_1s_{key}", 1, **kwargs)
    s1_results[key] = loss

s1_baseline = s1_results["baseline"]
s1_ranked = sorted(s1_results.items(), key=lambda x: x[1])

# ── STAGE 2: top 5 at 2 steps ─────────────────────────────────────────────
top5 = [k for k, _ in s1_ranked[:5]]
print(f"\n{'═'*60}")
print("STAGE 2 — 2s (2 steps) — top 5 promoted")
print("═" * 60)
s2_results = {}
for key in ["baseline"] + [k for k in top5 if k != "baseline"]:
    loss = run_config(f"tiered_2s_{key}", 2, **ALL_CONFIGS[key])
    s2_results[key] = loss

s2_baseline = s2_results["baseline"]
s2_ranked = sorted(s2_results.items(), key=lambda x: x[1])

# ── STAGE 3: top 3 at 3 steps ─────────────────────────────────────────────
top3 = [k for k, _ in s2_ranked[:3]]
print(f"\n{'═'*60}")
print("STAGE 3 — 3s (3 steps) — top 3 promoted")
print("═" * 60)
s3_results = {}
for key in (["baseline"] + [k for k in top3 if k != "baseline"]):
    loss = run_config(f"tiered_3s_{key}", 3, **ALL_CONFIGS[key])
    s3_results[key] = loss

s3_baseline = s3_results["baseline"]

# ── TABLES ────────────────────────────────────────────────────────────────
def decision(delta, promoted_keys, key):
    if key == "baseline": return "baseline"
    if key in promoted_keys: return "promote ✓"
    return "drop"

print("\n\n" + "═"*80)
print("### Stage 1 — 1s")
print(f"{'Run':<22} {'Dur':>4}  {'Architecture change':<26} {'Loss':>7} {'Base':>7} {'Delta':>7}  Decision")
print("-"*80)
promoted_s1 = set(top5)
for key, loss in s1_ranked:
    delta = loss - s1_baseline
    d = decision(delta, promoted_s1, key)
    print(f"tiered_1s_{key:<12} {'1s':>4}  {ARCH_DESC[key]:<26} {loss:>7.4f} {s1_baseline:>7.4f} {delta:>+7.4f}  {d}")

print("\n### Stage 2 — 2s")
print(f"{'Run':<22} {'Dur':>4}  {'Architecture change':<26} {'Loss':>7} {'Base':>7} {'Delta':>7}  Decision")
print("-"*80)
promoted_s2 = set(top3)
for key, loss in s2_ranked:
    delta = loss - s2_baseline
    d = decision(delta, promoted_s2, key)
    print(f"tiered_2s_{key:<12} {'2s':>4}  {ARCH_DESC[key]:<26} {loss:>7.4f} {s2_baseline:>7.4f} {delta:>+7.4f}  {d}")

print("\n### Stage 3 — 3s")
print(f"{'Run':<22} {'Dur':>4}  {'Architecture change':<26} {'Loss':>7} {'Base':>7} {'Delta':>7}  Decision")
print("-"*80)
for key, loss in sorted(s3_results.items(), key=lambda x: x[1]):
    delta = loss - s3_baseline
    d = "baseline" if key == "baseline" else ("finalist ✓" if loss <= s3_baseline else "drop")
    print(f"tiered_3s_{key:<12} {'3s':>4}  {ARCH_DESC[key]:<26} {loss:>7.4f} {s3_baseline:>7.4f} {delta:>+7.4f}  {d}")

print("\nDone.")
