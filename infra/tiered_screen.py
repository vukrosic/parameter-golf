"""Single-process tiered screening. Loads CUDA + data once, loops configs.

Usage:
    python3 infra/tiered_screen.py --screen screens/<topic>.py [--ladder quick|standard|thorough]
    python3 infra/tiered_screen.py  # ad-hoc: edit CONFIGS below

Ladder presets:
    quick     1 → 2 steps,   5 candidates → top 2  (seconds, default)
    standard  3 → 6 steps,   7 candidates → top 3  (minutes)
    thorough  10 → 20 steps, 10 candidates → top 5 (longer)

Screen config files live in screens/<topic>.py. Each must define a CONFIGS list:
    CONFIGS = [
        ("name", "one-line description of what it tests", {"override_key": value}),
        ...  # first entry is always the baseline with overrides={}
    ]
"""
import argparse, importlib.util, os, sys, time, json, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_mem_efficient_sdp, enable_math_sdp
enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
device = torch.device("cuda", 0)
torch.cuda.set_device(device)

import train_gpt as tg

# ── LADDER PRESETS ────────────────────────────────────────────────────────
LADDERS = {
    "quick":    dict(s1=1,  s2=2,  top1=5,  top2=2),
    "standard": dict(s1=3,  s2=6,  top1=7,  top2=3),
    "thorough": dict(s1=10, s2=20, top1=10, top2=5),
}

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--screen", default=None,
                    help="Path to screen config .py file defining CONFIGS list")
parser.add_argument("--ladder", default="quick", choices=LADDERS.keys())
parser.add_argument("--topic",  default=None,
                    help="Report topic label (defaults to screen filename stem)")
args = parser.parse_args()

ladder   = LADDERS[args.ladder]
S1_STEPS = ladder["s1"]
S2_STEPS = ladder["s2"]
TOP1     = ladder["top1"]
TOP2     = ladder["top2"]

# ── LOAD CONFIGS ──────────────────────────────────────────────────────────
if args.screen:
    screen_path = os.path.abspath(args.screen)
    spec = importlib.util.spec_from_file_location("screen_cfg", screen_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    CONFIGS = mod.CONFIGS
    _topic  = args.topic or os.path.splitext(os.path.basename(screen_path))[0]
else:
    # ── AD-HOC: edit freely for quick one-offs ────────────────────────────
    CONFIGS = [
        ("baseline", "Baseline — no changes.", {}),
        # ("my_variant", "One sentence description.", {"mlp_act": "swiglu"}),
    ]
    _topic = args.topic or "adhoc"

# ── MODEL CONFIG ──────────────────────────────────────────────────────────
SEED         = 1337
VOCAB_SIZE   = 1024
DATA_PATH    = "./data/datasets/fineweb10B_sp1024"
TRAIN_FILES  = os.path.join(DATA_PATH, "fineweb_train_*.bin")
GRAD_ACCUM   = 8
SEQ_LEN      = 256
BATCH_TOKENS = 32768

# Full competition model defaults
FULL = dict(
    num_layers=9, model_dim=512, num_heads=8, num_kv_heads=4,
    mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
    logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
    attnres_mode="none", mlp_act="relu2", act_power=2.0,
    act_gate_floor=0.5, embed_bottleneck=0, num_unique_blocks=0,
    num_cycles=1, conv_kernel=0, num_experts=0,
    resid_scale_init=1.0, stoch_depth_rate=0.0,
    highway_net=False, skip_weight_init=1.0,
)

print("Loading data...", flush=True)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
train_loader = tg.DistributedTokenLoader(TRAIN_FILES, 0, 1, device)
print(f"Data loaded. Running screen '{_topic}' with ladder '{args.ladder}' "
      f"({S1_STEPS}→{S2_STEPS} steps, {len(CONFIGS)} candidates → top {TOP2}).\n", flush=True)

# ── TRAIN LOOP ────────────────────────────────────────────────────────────
def run_config(name, steps, arch_desc, overrides):
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    cfg   = {**FULL, **overrides}
    model = tg.GPT(vocab_size=VOCAB_SIZE, **cfg).to(device).bfloat16()
    tg.restore_low_dim_params_to_fp32(model)
    n_params = sum(p.numel() for p in model.parameters())
    opt   = torch.optim.Adam(model.parameters(), lr=0.04)
    model.train()
    losses, t0 = [], time.perf_counter()
    grad_scale = 1.0 / GRAD_ACCUM
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(GRAD_ACCUM):
            x, y = train_loader.next_batch(BATCH_TOKENS, SEQ_LEN, GRAD_ACCUM)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            (loss * grad_scale).backward()
            step_loss += loss.item() * grad_scale
        opt.step()
        losses.append(step_loss)
    elapsed  = time.perf_counter() - t0
    avg_loss = sum(losses) / len(losses)
    os.makedirs(f"results/{name}", exist_ok=True)
    with open(f"results/{name}/summary.json", "w") as f:
        json.dump({"run_id": name, "steps": steps, "elapsed_s": round(elapsed, 2),
                   "train_loss": round(avg_loss, 4), "n_params": n_params,
                   "arch": arch_desc, "overrides": overrides}, f, indent=2)
    print(f"  {name:<32}  loss={avg_loss:.4f}  params={n_params:,}  ({elapsed:.1f}s)", flush=True)
    return avg_loss

# ── STAGE 1 ───────────────────────────────────────────────────────────────
print("═"*60)
print(f"STAGE 1 — {S1_STEPS} step(s) — {len(CONFIGS)} candidates")
print("═"*60)
s1, BASELINE_KEY = {}, CONFIGS[0][0]
for name, desc, ov in CONFIGS:
    s1[name] = (run_config(f"s1_{_topic}_{name}", S1_STEPS, desc, ov), desc, ov)

s1_baseline_loss = s1[BASELINE_KEY][0]
s1_ranked        = sorted(s1.items(), key=lambda x: x[1][0])
top_names        = [k for k, _ in s1_ranked if k != BASELINE_KEY][:TOP2]

# ── STAGE 2 ───────────────────────────────────────────────────────────────
promote = [BASELINE_KEY] + top_names
print(f"\n{'═'*60}")
print(f"STAGE 2 — {S2_STEPS} step(s) — top {TOP2} promoted: {top_names}")
print("═"*60)
s2 = {}
cfg_map = {c[0]: c for c in CONFIGS}
for name in promote:
    _, desc, ov = cfg_map[name]
    s2[name] = (run_config(f"s2_{_topic}_{name}", S2_STEPS, desc, ov), desc)

s2_baseline_loss = s2[BASELINE_KEY][0]
s2_ranked        = sorted(s2.items(), key=lambda x: x[1][0])

# ── REPORT ────────────────────────────────────────────────────────────────
date        = datetime.date.today().strftime("%Y%m%d")
report_path = f"results/tiered_screen_{_topic}_{date}.md"

s2_winners = [(k, l, d) for k, (l, d) in s2_ranked if l < s2_baseline_loss and k != BASELINE_KEY]
s2_flipped = [(k, l, d) for k, (l, d) in s2_ranked if l >= s2_baseline_loss and k != BASELINE_KEY]

lines = [
    f"# Tiered Screen — {_topic} — {date}",
    "",
    f"**Model:** full competition model (9L × 512d × 8/4 heads, MLP_MULT=2)  ",
    f"**Ladder:** `{args.ladder}` — {S1_STEPS} step(s) → {S2_STEPS} step(s) | "
    f"{len(CONFIGS)} candidates → top {TOP2}  ",
    "**Optimizer:** plain Adam (no Muon/compile). Loss values are relative — compare within stage only.",
    "",
]
if hasattr(mod if args.screen else type('_', (), {}), "WHY"):
    lines += [f"**Why these candidates:** {mod.WHY}", ""]
lines += [
    "---",
    "",
    f"### Stage 1 — {S1_STEPS} step(s): screen all {len(CONFIGS)} candidates",
    "",
    "| Run | What it tests | Loss | Baseline | Delta | Decision |",
    "|---|---|---:|---:|---:|---|",
]
for k, (loss, desc, _) in s1_ranked:
    delta = loss - s1_baseline_loss
    if k == BASELINE_KEY:          dec = "baseline"
    elif k in top_names:           dec = "promote ✓"
    else:                          dec = "drop"
    lines.append(f"| `{k}` | {desc} | {loss:.4f} | {s1_baseline_loss:.4f} | {delta:+.4f} | {dec} |")

lines += [
    "",
    f"Promoted to stage 2: {', '.join(f'**{k}**' for k in top_names)}",
    "",
    "---",
    "",
    f"### Stage 2 — {S2_STEPS} step(s): confirm or expose noise",
    "",
    "| Run | What it tests | Loss | Baseline | Delta | Decision |",
    "|---|---|---:|---:|---:|---|",
]
for k, (loss, desc) in s2_ranked:
    delta = loss - s2_baseline_loss
    if k == BASELINE_KEY:          dec = "baseline"
    elif loss < s2_baseline_loss:  dec = "finalist ✓"
    else:                          dec = "drop"
    lines.append(f"| `{k}` | {desc} | {loss:.4f} | {s2_baseline_loss:.4f} | {delta:+.4f} | {dec} |")

lines += ["", "---", "", "## What happened", ""]
if s2_winners:
    wnames = ", ".join(f"`{k}` ({d})" for k, _, d in s2_winners)
    lines += [
        f"**Survived both stages:** {wnames}.",
        "Ranking was consistent — this is signal, not noise. Promote to a 500-step explore run.",
    ]
else:
    lines.append("**Nothing beat the baseline at stage 2.** Stage-1 gains were initialization-transient. Drop everything and try a different direction.")
if s2_flipped:
    fnames = ", ".join(f"`{k}`" for k, _, _ in s2_flipped)
    lines.append(f"**Flipped negative at stage 2:** {fnames} — fast-start artifact, not real.")
lines += [
    "",
    f"_Baseline loss rises {s1_baseline_loss:.4f} → {s2_baseline_loss:.4f} between stages "
    "(Adam without warmup climbs briefly before descending — normal, compare within stage only)._",
]

report = "\n".join(lines)
os.makedirs("results", exist_ok=True)
with open(report_path, "w") as f:
    f.write(report)

print(f"\n{'═'*70}")
print(report)
print(f"\nReport written: {report_path}")
