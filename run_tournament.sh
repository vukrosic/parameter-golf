#!/bin/bash
# Tournament-style architecture search with scaling law validation
# Round 1: 30s runs (batch=65536, ~120 steps) — cast wide net, 30 configs
# Round 2: 2min runs (batch=131072, ~240 steps) — top 10 survivors
# Round 3: 8min runs (batch=524288, 500 steps) — top 5, validate scaling
# Round 4: 30min runs (batch=524288, 1000 steps) — top 2 × recipe combos
#
# Total time: ~23min + ~25min + ~80min + ~4h = ~6h
# But Round 1 alone gives us massive signal in 23 minutes!
set -uo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0

mkdir -p results/tournament

# Generic run function: run NAME BATCH ITERS CSV_FILE CONFIG...
# Saves config to .cfg file for later rounds to re-read
run_exp() {
    local name="$1"; shift
    local batch="$1"; shift
    local iters="$1"; shift
    local csv="$1"; shift
    local config="$*"

    echo "--- $name (${iters}steps, batch=${batch}) ---"
    mkdir -p "results/tournament/$name"
    # Save config for later rounds
    echo "$config" > "results/tournament/$name/config.cfg"

    eval "export TRAIN_BATCH_TOKENS=$batch MAX_WALLCLOCK_SECONDS=0 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=50 ITERATIONS=$iters $config RUN_ID=t_$name"
    if python3 train_gpt_golf.py 2>&1 | tee "results/tournament/$name/train.log"; then
        local int8_bpb=$(grep "final_int8_zlib_roundtrip_exact" "results/tournament/$name/train.log" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
        local step_ms=$(grep "step_avg" "results/tournament/$name/train.log" 2>/dev/null | tail -1 | grep -oP 'step_avg:\K[0-9.]+' || echo "N/A")
        local params=$(grep "model_params" "results/tournament/$name/train.log" 2>/dev/null | grep -oP 'model_params:\K[0-9]+' || echo "N/A")
        local size=$(grep "Serialized model int8.zlib" "results/tournament/$name/train.log" 2>/dev/null | grep -oP '[0-9]+ bytes' | head -1 | grep -oP '[0-9]+' || echo "N/A")
        echo "$name,$iters,$batch,$int8_bpb,$step_ms,$params,$size" >> "$csv"
        echo ">>> $name: int8=$int8_bpb step=${step_ms}ms params=$params size=${size}B"
    else
        echo "$name,$iters,$batch,CRASH,CRASH,CRASH,CRASH" >> "$csv"
        echo ">>> $name: CRASHED"
    fi
    sleep 1
}

# Promote top N from a CSV, re-run at new scale
promote() {
    local src_csv="$1"; shift
    local dst_csv="$1"; shift
    local topn="$1"; shift
    local batch="$1"; shift
    local iters="$1"; shift
    local prefix="$1"; shift

    local winners=$(tail -n +2 "$src_csv" | grep -v CRASH | grep -v N/A | sort -t',' -k4 -n | head -"$topn" | cut -d',' -f1)
    for name in $winners; do
        local cfg_file="results/tournament/$name/config.cfg"
        if [ ! -f "$cfg_file" ]; then echo "SKIP $name: no config"; continue; fi
        local config=$(cat "$cfg_file")
        local new_name="${prefix}_${name#*_}"
        run_exp "$new_name" "$batch" "$iters" "$dst_csv" $config
    done
}

# ============================================================
# ROUND 1: 30-second sprints (~120 steps each, batch=65536)
# ~30 configs × 45s = ~23 min total
# ============================================================
R1_CSV="results/tournament/round1.csv"
echo "name,iters,batch,int8_bpb,step_ms,params,size_bytes" > "$R1_CSV"

echo ""
echo "============================================"
echo "ROUND 1: 30-second sprints (120 steps, batch=65536)"
echo "============================================"

# --- Group A: Recurrence depth/width sweep (no SwiGLU) ---
run_exp "r1_baseline"     65536 120 "$R1_CSV" NUM_LAYERS=9  MODEL_DIM=512 NUM_HEADS=8  NUM_KV_HEADS=4 MLP_MULT=2
run_exp "r1_3x3_512"      65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=3 NUM_EFFECTIVE_LAYERS=9  MODEL_DIM=512 NUM_HEADS=8  NUM_KV_HEADS=4 MLP_MULT=2
run_exp "r1_4x4_640"      65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2
run_exp "r1_3x6_704"      65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=3 NUM_EFFECTIVE_LAYERS=18 MODEL_DIM=704 NUM_HEADS=11 NUM_KV_HEADS=1 MLP_MULT=2
run_exp "r1_5x3_576"      65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=5 NUM_EFFECTIVE_LAYERS=15 MODEL_DIM=576 NUM_HEADS=9  NUM_KV_HEADS=3 MLP_MULT=2
run_exp "r1_6x3_512"      65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=6 NUM_EFFECTIVE_LAYERS=18 MODEL_DIM=512 NUM_HEADS=8  NUM_KV_HEADS=4 MLP_MULT=2
run_exp "r1_4x5_608"      65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=20 MODEL_DIM=608 NUM_HEADS=8  NUM_KV_HEADS=4 MLP_MULT=2
run_exp "r1_3x8_672"      65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=3 NUM_EFFECTIVE_LAYERS=24 MODEL_DIM=672 NUM_HEADS=8  NUM_KV_HEADS=4 MLP_MULT=2
run_exp "r1_4x6_640"      65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=24 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2
run_exp "r1_2x8_512"      65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=2 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=512 NUM_HEADS=8  NUM_KV_HEADS=4 MLP_MULT=2
run_exp "r1_2x12_640"     65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=2 NUM_EFFECTIVE_LAYERS=24 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2
run_exp "r1_5x4_576"      65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=5 NUM_EFFECTIVE_LAYERS=20 MODEL_DIM=576 NUM_HEADS=9  NUM_KV_HEADS=3 MLP_MULT=2
run_exp "r1_3x10_704"     65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=3 NUM_EFFECTIVE_LAYERS=30 MODEL_DIM=704 NUM_HEADS=11 NUM_KV_HEADS=1 MLP_MULT=2
run_exp "r1_4x8_640"      65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=32 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2

# --- Group B: SwiGLU on key configs ---
run_exp "r1_sw_baseline"   65536 120 "$R1_CSV" NUM_LAYERS=9  MODEL_DIM=512 NUM_HEADS=8  NUM_KV_HEADS=4 MLP_MULT=2 USE_SWIGLU=1
run_exp "r1_sw_4x4_640"    65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1
run_exp "r1_sw_3x6_704"    65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=3 NUM_EFFECTIVE_LAYERS=18 MODEL_DIM=704 NUM_HEADS=11 NUM_KV_HEADS=1 MLP_MULT=2 USE_SWIGLU=1
run_exp "r1_sw_4x6_640"    65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=24 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1
run_exp "r1_sw_3x8_672"    65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=3 NUM_EFFECTIVE_LAYERS=24 MODEL_DIM=672 NUM_HEADS=8  NUM_KV_HEADS=4 MLP_MULT=2 USE_SWIGLU=1
run_exp "r1_sw_4x8_640"    65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=32 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=2 USE_SWIGLU=1

# --- Group C: Wider models (fill 16MB budget) ---
run_exp "r1_sw_4x4_768"    65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=4 MLP_MULT=2 USE_SWIGLU=1
run_exp "r1_sw_3x4_832"    65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=3 NUM_EFFECTIVE_LAYERS=12 MODEL_DIM=832 NUM_HEADS=13 NUM_KV_HEADS=1 MLP_MULT=2 USE_SWIGLU=1
run_exp "r1_sw_5x3_768"    65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=5 NUM_EFFECTIVE_LAYERS=15 MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=4 MLP_MULT=2 USE_SWIGLU=1
run_exp "r1_sw_2x8_768"    65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=2 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=4 MLP_MULT=2 USE_SWIGLU=1

# --- Group D: MLP width & KV heads ---
run_exp "r1_4x4_640_m3"    65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=3
run_exp "r1_sw_4x4_640_m3" 65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=3 USE_SWIGLU=1
run_exp "r1_sw_4x4_640_k1" 65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=1  MLP_MULT=2 USE_SWIGLU=1
run_exp "r1_sw_4x4_640_k10" 65536 120 "$R1_CSV" NUM_UNIQUE_BLOCKS=4 NUM_EFFECTIVE_LAYERS=16 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=10 MLP_MULT=2 USE_SWIGLU=1

echo ""
echo "============================================"
echo "ROUND 1 COMPLETE — 30 configs tested"
echo "============================================"
sort -t',' -k4 -n "$R1_CSV" | column -t -s','

# ============================================================
# ROUND 2: 2-minute runs (~240 steps, batch=131072)
# Promote top 10 from Round 1
# ============================================================
R2_CSV="results/tournament/round2.csv"
echo "name,iters,batch,int8_bpb,step_ms,params,size_bytes" > "$R2_CSV"

echo ""
echo "============================================"
echo "ROUND 2: 2-min runs (240 steps, batch=131072)"
echo "Promoting top 10 from Round 1"
echo "============================================"

promote "$R1_CSV" "$R2_CSV" 10 131072 240 "r2"

echo ""
echo "============================================"
echo "ROUND 2 COMPLETE"
echo "============================================"
sort -t',' -k4 -n "$R2_CSV" | column -t -s','

# ============================================================
# ROUND 3: 8-minute runs (500 steps, batch=524288)
# Promote top 5 from Round 2
# ============================================================
R3_CSV="results/tournament/round3.csv"
echo "name,iters,batch,int8_bpb,step_ms,params,size_bytes" > "$R3_CSV"

echo ""
echo "============================================"
echo "ROUND 3: 8-min full-batch runs (500 steps, batch=524288)"
echo "Promoting top 5 from Round 2"
echo "============================================"

promote "$R2_CSV" "$R3_CSV" 5 524288 500 "r3"

echo ""
echo "============================================"
echo "ROUND 3 COMPLETE"
echo "============================================"
sort -t',' -k4 -n "$R3_CSV" | column -t -s','

# ============================================================
# ROUND 4: 30-min runs (1000 steps) — top 2 × recipe variations
# ============================================================
R4_CSV="results/tournament/round4.csv"
echo "name,iters,batch,int8_bpb,step_ms,params,size_bytes" > "$R4_CSV"

echo ""
echo "============================================"
echo "ROUND 4: 30-min runs (1000 steps) + recipe variations"
echo "Top 2 from Round 3 × 4 recipe combos = 8 runs"
echo "============================================"

# Get top 2 configs
TOP2=$(tail -n +2 "$R3_CSV" | grep -v CRASH | grep -v N/A | sort -t',' -k4 -n | head -2 | cut -d',' -f1)

for name in $TOP2; do
    cfg="results/tournament/$name/config.cfg"
    if [ ! -f "$cfg" ]; then continue; fi
    config=$(cat "$cfg")
    base="${name#r3_}"

    # Plain
    run_exp "r4_${base}_plain" 524288 1000 "$R4_CSV" $config
    # QAT 50%
    run_exp "r4_${base}_qat50" 524288 1000 "$R4_CSV" $config QAT_START_FRAC=0.5
    # QAT 30%
    run_exp "r4_${base}_qat30" 524288 1000 "$R4_CSV" $config QAT_START_FRAC=0.3
    # Higher LR
    run_exp "r4_${base}_lr15"  524288 1000 "$R4_CSV" $config TIED_EMBED_LR=0.075 MATRIX_LR=0.06 SCALAR_LR=0.06
done

echo ""
echo "============================================"
echo "ROUND 4 COMPLETE"
echo "============================================"
sort -t',' -k4 -n "$R4_CSV" | column -t -s','

# ============================================================
# FINAL SUMMARY
# ============================================================
echo ""
echo "========================================================"
echo "FULL TOURNAMENT COMPLETE — SCALING LAW ANALYSIS"
echo "========================================================"
echo ""
echo "--- R1 (30s, 65K batch, 120 steps) ---"
sort -t',' -k4 -n "$R1_CSV" | column -t -s','
echo ""
echo "--- R2 (2min, 131K batch, 240 steps) ---"
sort -t',' -k4 -n "$R2_CSV" | column -t -s','
echo ""
echo "--- R3 (8min, 524K batch, 500 steps) ---"
sort -t',' -k4 -n "$R3_CSV" | column -t -s','
echo ""
echo "--- R4 (30min, 524K batch, 1000 steps + recipes) ---"
sort -t',' -k4 -n "$R4_CSV" | column -t -s','
echo ""
echo "WINNER:"
tail -n +2 "$R4_CSV" | grep -v CRASH | sort -t',' -k4 -n | head -1 | column -t -s','
