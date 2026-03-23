#!/usr/bin/env python3
"""
Visual queue runner for tiny (~3M param) ablation experiments on 3090.

Usage:
    python3 infra/run_queue_tiny.py                             # run all 100
    python3 infra/run_queue_tiny.py --test                      # run ONLY first exp, check timing
    python3 infra/run_queue_tiny.py --verify                    # test timing + test wallclock cap
    python3 infra/run_queue_tiny.py --dry-run                   # print plan, no execution
    python3 infra/run_queue_tiny.py --start 15                  # resume from exp #15
    python3 infra/run_queue_tiny.py queues/other.txt            # custom queue file

Expected per-experiment: ~100s (1.7min)
Hard limits enforced:
  1. MAX_WALLCLOCK_SECONDS=90  → train_gpt.py stops training cleanly, runs final eval
  2. OUTER_TIMEOUT_S=145       → Python subprocess kill if train_gpt.py hangs
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ── Timing thresholds ─────────────────────────────────────────────────────────
# 300 steps × 157ms = 47s training
# + final val (VAL_BATCH_SIZE=4194304, B=512 seqs, ~15s) + overhead = ~70s
# INT8 eval is SKIPPED: runner kills process once val_bpb appears in log
EXPECTED_S       = 70    # expected wall time per experiment (seconds)
WARN_S           = 95    # yellow warning: running slow
OVER_S           = 120   # red: clearly overtime
OUTER_TIMEOUT    = 150   # hard kill fallback if process hangs
KILL_AFTER_VAL_S = 4     # seconds to wait after val_bpb logged before killing (skip int8)

# ── ANSI ─────────────────────────────────────────────────────────────────────
R = "\033[0m"
BOLD = "\033[1m"; DIM = "\033[2m"
RED = "\033[31m"; YLW = "\033[33m"; GRN = "\033[32m"; CYN = "\033[36m"; MAG = "\033[35m"

def c(code, text): return f"{code}{text}{R}"

REPO = Path(__file__).parent.parent


# ── Queue parsing ─────────────────────────────────────────────────────────────

def parse_queue(path: str) -> list[dict]:
    exps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name, steps_str = parts[0], parts[1]
            env = {}
            for p in parts[2:]:
                if p.startswith('#'):
                    break
                k, _, v = p.partition('=')
                if k:
                    env[k] = v
            exps.append({"name": name, "steps": int(steps_str), "env": env})
    return exps


def is_done(name: str) -> bool:
    return (REPO / "results" / name / "train.log").exists()


# ── Log parsing ───────────────────────────────────────────────────────────────

def _last_match(path: str, pattern: str) -> Optional[re.Match]:
    try:
        lines = Path(path).read_text(errors='replace').splitlines()
        for line in reversed(lines):
            m = re.search(pattern, line)
            if m:
                return m
    except FileNotFoundError:
        pass
    return None


def get_val_bpb(log_path: str) -> Optional[float]:
    m = _last_match(log_path, r'val_bpb:([\d.]+)')
    return float(m.group(1)) if m else None


def get_last_step(log_path: str) -> tuple[Optional[int], Optional[int]]:
    # Use negative lookbehind to skip "warmup_step:" lines (warmup_ precedes step:)
    m = _last_match(log_path, r'(?<![a-z_])step:(\d+)/(\d+)')
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def get_step_avg_ms(log_path: str) -> Optional[float]:
    """Latest step_avg from log (ms per training step)."""
    m = _last_match(log_path, r'step_avg:([\d.]+)ms')
    return float(m.group(1)) if m else None


def detect_failure(log_path: str) -> Optional[str]:
    """Return a short error description if the log shows a crash."""
    try:
        text = Path(log_path).read_text(errors='replace')
        if 'nan' in text.lower() or 'NaN' in text:
            return "NaN in loss"
        if 'RuntimeError' in text or 'CUDA error' in text:
            for line in text.splitlines():
                if 'RuntimeError' in line or 'CUDA error' in line:
                    return line.strip()[:80]
    except FileNotFoundError:
        pass
    return None


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_experiment(
    exp: dict,
    log_path: str,
    display_idx: int,
    total: int,
) -> dict:
    """
    Run one experiment. Prints live status. Returns result dict.
    """
    name = exp["name"]
    steps = exp["steps"]
    env = {**os.environ, **exp["env"]}

    (REPO / "results" / name).mkdir(parents=True, exist_ok=True)
    (REPO / "logs").mkdir(exist_ok=True)

    cmd = ["bash", "infra/run_experiment.sh", name, str(steps)]

    # Header
    print(f"\n{c(BOLD, f'[{display_idx:02d}/{total}]')} {c(CYN, name)}")
    BASE_DEFS = {
        "NUM_LAYERS":"6","MODEL_DIM":"256","NUM_HEADS":"4","NUM_KV_HEADS":"2",
        "MLP_MULT":"2","TRAIN_BATCH_TOKENS":"65536","VAL_LOSS_EVERY":"0",
        "WARMUP_STEPS":"10","WARMDOWN_ITERS":"50","MAX_WALLCLOCK_SECONDS":"75",
        "VAL_BATCH_SIZE":"4194304",
    }
    extras = {k: v for k, v in exp["env"].items() if BASE_DEFS.get(k) != v}
    if extras:
        print(f"  {c(DIM, ' '.join(f'{k}={v}' for k, v in extras.items()))}")
    else:
        print(f"  {c(DIM, '(base config)')}")

    start = time.time()
    timed_out = False
    killed_after_val = False

    with open(log_path, 'w') as log_f:
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=log_f, stderr=subprocess.STDOUT,
            cwd=str(REPO)
        )

    POLL = 3  # seconds between status polls
    warned_overtime = False
    val_done_at: Optional[float] = None   # time when val_bpb first appeared

    while True:
        try:
            proc.wait(timeout=POLL)
            break  # finished naturally
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start

            if elapsed >= OUTER_TIMEOUT:
                proc.kill()
                proc.wait()
                timed_out = True
                break

            # ── Early kill: skip int8 eval ─────────────────────────────────
            # Once training is done AND val_bpb is logged, kill to skip int8.
            cur_step, tot_step = get_last_step(log_path)
            training_done = (cur_step is not None and cur_step == tot_step)
            if training_done and val_done_at is None and get_val_bpb(log_path) is not None:
                val_done_at = time.time()
            if val_done_at is not None and (time.time() - val_done_at) >= KILL_AFTER_VAL_S:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                killed_after_val = True
                break  # cleanly killed after val

            # Build status line
            elapsed_str = f"{elapsed:.0f}s"

            if elapsed >= OVER_S:
                if not warned_overtime:
                    warned_overtime = True
                    print(f"\n  {c(RED, '⚠ OVERTIME')} — experiment is running longer than {OVER_S}s")
                elapsed_color = c(RED, f"⚠ {elapsed_str}")
            elif elapsed >= WARN_S:
                elapsed_color = c(YLW, f"~ {elapsed_str}")
            else:
                elapsed_color = elapsed_str

            step_info = ""
            if cur_step is not None:
                pct = cur_step / tot_step * 100 if tot_step else 0
                step_info = f"  step {cur_step}/{tot_step} ({pct:.0f}%)"

            bar_done = int(elapsed / EXPECTED_S * 20)
            bar_done = min(bar_done, 20)
            bar = "█" * bar_done + "░" * (20 - bar_done)
            bar_color = c(RED if elapsed >= OVER_S else (YLW if elapsed >= WARN_S else GRN), bar)

            print(f"  ▶ [{bar_color}] {elapsed_color}{step_info}    ", end='\r', flush=True)

    elapsed = time.time() - start
    val_bpb = get_val_bpb(log_path)
    cur_step, tot_step = get_last_step(log_path)
    step_avg = get_step_avg_ms(log_path)
    exit_code = proc.returncode
    # Don't flag as failure if we intentionally killed after val_bpb
    if killed_after_val:
        exit_code = 0
    failure = detect_failure(log_path) if exit_code != 0 else None

    # Final result line
    print(" " * 70, end='\r')  # clear progress line

    if timed_out:
        status = c(RED, f"TIMEOUT ({elapsed:.0f}s — subprocess killed)")
    elif exit_code != 0:
        err = failure or f"exit={exit_code}"
        status = c(RED, f"FAILED ({err}) in {elapsed:.0f}s")
    elif val_bpb is None:
        status = c(YLW, f"NO VAL_BPB in {elapsed:.0f}s")
    elif elapsed >= OVER_S:
        status = c(YLW, f"SLOW {elapsed:.0f}s (>{OVER_S}s expected)")
    else:
        status = c(GRN, f"✓ {elapsed:.0f}s")

    bpb_str = f"val_bpb={c(BOLD, f'{val_bpb:.4f}')}" if val_bpb else "val_bpb=N/A"
    step_str = (f"steps={cur_step}/{tot_step}" if cur_step else "")
    ms_str = (f"  {step_avg:.0f}ms/step" if step_avg else "")

    print(f"  {status} | {bpb_str} | {step_str}{ms_str}")

    return {
        "name": name,
        "elapsed": elapsed,
        "val_bpb": val_bpb,
        "step_done": cur_step,
        "step_total": tot_step,
        "step_avg_ms": step_avg,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "failure": failure,
    }


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(results: list[dict], prefix=""):
    if not results:
        return
    ok = [r for r in results if r["val_bpb"] is not None]
    failed = [r for r in results if r["val_bpb"] is None]
    slow = [r for r in results if r["elapsed"] >= WARN_S and r["val_bpb"] is not None]
    timed = [r for r in results if r["timed_out"]]

    print(f"\n  {c(DIM, '─' * 62)}")
    print(f"  {c(BOLD, prefix + 'Summary:')}")
    print(f"  Done: {len(results)}  |  "
          f"OK: {c(GRN, str(len(ok)))}  |  "
          f"Failed: {c(RED, str(len(failed)))}  |  "
          f"Slow: {c(YLW, str(len(slow)))}  |  "
          f"Timeout: {c(RED, str(len(timed)))}")

    if ok:
        best = min(ok, key=lambda r: r["val_bpb"])
        worst = max(ok, key=lambda r: r["val_bpb"])
        avg_time = sum(r["elapsed"] for r in results) / len(results)
        best_bpb = f"{best['val_bpb']:.4f}"
        worst_bpb = f"{worst['val_bpb']:.4f}"
        print(f"  Best:  {c(GRN, best_bpb)} — {best['name']}")
        print(f"  Worst: {c(YLW, worst_bpb)} — {worst['name']}")
        print(f"  Avg time: {avg_time:.0f}s/exp | Total: {sum(r['elapsed'] for r in results)/60:.1f}min")

    if failed:
        print(f"  {c(RED, 'Failures:')} " + ", ".join(r['name'] for r in failed[:5]))

    print(f"  {c(DIM, '─' * 62)}")


def print_top_n(results: list[dict], n=10):
    ok = sorted([r for r in results if r["val_bpb"] is not None], key=lambda r: r["val_bpb"])
    if not ok:
        return
    print(f"\n  {c(BOLD, f'Top {min(n, len(ok))} by val_bpb:')}")
    for i, r in enumerate(ok[:n], 1):
        steps_str = f"  ({r['step_done']}/{r['step_total']} steps)" if r['step_done'] else ""
        bpb_s = f"{r['val_bpb']:.4f}"
        print(f"  {i:2d}. {c(GRN, bpb_s)}  {r['name']:<30}{steps_str}")


# ── Verify mode ───────────────────────────────────────────────────────────────

def run_verify(queue_file: str):
    """
    Two-part verification:
      Test A: timing — baseline should finish in ~100s
      Test B: wallclock cap — MAX_WALLCLOCK_SECONDS=5 should stop early with val_bpb reported
    """
    print(c(BOLD, "\n══════════════════════════════════════════════════════════════"))
    print(c(BOLD, "  VERIFICATION MODE"))
    print(c(BOLD, "══════════════════════════════════════════════════════════════"))

    exps = parse_queue(queue_file)
    baseline = exps[0]

    # ── Test A: timing ──────────────────────────────────────────────────────
    print(f"\n{c(BOLD, 'Test A: Timing')} — expect ~{EXPECTED_S}s for baseline")
    print(f"  Config: {baseline['env'].get('NUM_LAYERS','?')}L "
          f"{baseline['env'].get('MODEL_DIM','?')}d "
          f"batch={baseline['env'].get('TRAIN_BATCH_TOKENS','?')} "
          f"steps={baseline['steps']}  MAX_WALLCLOCK={baseline['env'].get('MAX_WALLCLOCK_SECONDS','?')}s")

    r_a = run_experiment(baseline, "logs/verify_timing.txt", 1, 2)

    if r_a["timed_out"]:
        print(c(RED, "  ✗ OUTER TIMEOUT — something is very wrong (hung process)"))
    elif r_a["val_bpb"] is None:
        print(c(RED, "  ✗ No val_bpb — training script may have crashed"))
    elif r_a["elapsed"] > WARN_S:
        print(c(YLW, f"  ⚠ Slow: {r_a['elapsed']:.0f}s (expected ~{EXPECTED_S}s). "
                     f"100 exps would take ~{100*r_a['elapsed']/3600:.1f}hr"))
    elif r_a["elapsed"] < 60:
        print(c(YLW, f"  ⚠ Very fast: {r_a['elapsed']:.0f}s — might be using cached results?"))
    else:
        print(c(GRN, f"  ✓ Timing OK: {r_a['elapsed']:.0f}s  "
                     f"| 100 exps ≈ {100*r_a['elapsed']/3600:.1f}hr"))
        if r_a["step_avg_ms"]:
            print(f"    Step speed: {r_a['step_avg_ms']:.0f}ms/step "
                  f"(expected ~161ms)")

    # ── Test B: wallclock cap ───────────────────────────────────────────────
    print(f"\n{c(BOLD, 'Test B: Wallclock cap')} — MAX_WALLCLOCK_SECONDS=5 should stop early")
    print("  Training at ~161ms/step → should reach ~30 steps before cap triggers")

    cap_exp = {
        "name": "verify_wcap_test",
        "steps": 500,
        "env": {
            **baseline["env"],
            "MAX_WALLCLOCK_SECONDS": "5",
        }
    }

    r_b = run_experiment(cap_exp, "logs/verify_wcap.txt", 2, 2)

    if r_b["timed_out"]:
        print(c(RED, "  ✗ OUTER TIMEOUT — wallclock cap did not stop training!"))
    elif r_b["val_bpb"] is None:
        print(c(RED, "  ✗ No val_bpb after early stop — final eval did not run"))
    elif r_b["step_done"] is not None and r_b["step_done"] >= 490:
        print(c(YLW, f"  ⚠ Ran {r_b['step_done']}/500 steps — cap may not have triggered "
                     f"(total time: {r_b['elapsed']:.0f}s)"))
    elif r_b["step_done"] is not None and r_b["step_done"] < 100:
        print(c(GRN, f"  ✓ Wallclock cap works: stopped at step {r_b['step_done']}/500 "
                     f"in {r_b['elapsed']:.0f}s | val_bpb={r_b['val_bpb']:.4f}"))
    else:
        print(c(GRN, f"  ✓ Stopped at step {r_b['step_done']}/500 "
                     f"in {r_b['elapsed']:.0f}s | val_bpb={r_b['val_bpb']:.4f}"))

    # ── Verdict ─────────────────────────────────────────────────────────────
    timing_ok = r_a["val_bpb"] is not None and not r_a["timed_out"] and r_a["elapsed"] < WARN_S
    cap_ok = r_b["val_bpb"] is not None and not r_b["timed_out"] and (
        r_b["step_done"] is None or r_b["step_done"] < 200
    )

    print()
    if timing_ok and cap_ok:
        print(c(GRN, "  ✓ ALL CHECKS PASSED — ready to run full queue"))
        print(f"    Run: python3 infra/run_queue_tiny.py {queue_file}")
    else:
        if not timing_ok:
            print(c(YLW, "  ⚠ Timing check: issues detected — review before full run"))
        if not cap_ok:
            print(c(RED, "  ✗ Wallclock cap: NOT working as expected!"))


# ── Dry run ───────────────────────────────────────────────────────────────────

def run_dry(exps: list[dict]):
    skip = sum(1 for e in exps if is_done(e["name"]))
    run = len(exps) - skip
    print(c(BOLD, f"\n  DRY RUN — {len(exps)} experiments | {skip} done | {run} to run"))
    print(f"  Est time: ~{run * EXPECTED_S / 60:.0f}min (~{run * EXPECTED_S / 3600:.1f}hr)\n")

    BASE_DEFAULTS = {
        "NUM_LAYERS": "6", "MODEL_DIM": "256", "NUM_HEADS": "4", "NUM_KV_HEADS": "2",
        "MLP_MULT": "2", "TRAIN_BATCH_TOKENS": "65536", "VAL_LOSS_EVERY": "500",
        "WARMUP_STEPS": "10", "WARMDOWN_ITERS": "50", "MAX_WALLCLOCK_SECONDS": "90",
    }

    for i, exp in enumerate(exps, 1):
        done_marker = c(GRN, "✓") if is_done(exp["name"]) else c(DIM, "○")
        diffs = {k: v for k, v in exp["env"].items() if BASE_DEFAULTS.get(k) != v}
        extras_str = " ".join(f"{k}={v}" for k, v in diffs.items()) or c(DIM, "(base)")
        print(f"  {done_marker} [{i:03d}] {exp['name']:<30}  {extras_str}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Visual queue runner for tiny-model ablations"
    )
    ap.add_argument("queue_file", nargs="?", default="queues/tiny100_diverse.txt")
    ap.add_argument("--test",     action="store_true", help="Run first experiment only")
    ap.add_argument("--verify",   action="store_true", help="Run timing + wallclock cap tests")
    ap.add_argument("--dry-run",  action="store_true", help="Print plan, no execution")
    ap.add_argument("--start",    type=int, default=1,  help="Resume from experiment N (1-indexed)")
    ap.add_argument("--only",     type=str, default=None, help="Run only this experiment name")
    ap.add_argument("--summary-every", type=int, default=10)
    args = ap.parse_args()

    os.chdir(REPO)

    exps = parse_queue(args.queue_file)
    total = len(exps)
    skip_n = sum(1 for e in exps if is_done(e["name"]))

    # ── Banner ─────────────────────────────────────────────────────────────
    print(c(BOLD, "\n╔══════════════════════════════════════════════════════════════╗"))
    print(c(BOLD,  "║  TINY100 QUEUE RUNNER                                        ║"))
    print(c(BOLD,  "╚══════════════════════════════════════════════════════════════╝"))
    print(f"  Queue : {args.queue_file}")
    print(f"  Total : {total} experiments | {c(GRN, str(skip_n))} done | "
          f"{c(CYN, str(total - skip_n))} to run")
    print(f"  Est   : ~{(total-skip_n)*EXPECTED_S/60:.0f}min "
          f"(~{(total-skip_n)*EXPECTED_S/3600:.1f}hr)")
    print(f"  Limits: MAX_WALLCLOCK=75s (training) | kill after val_bpb (skip int8) | outer={OUTER_TIMEOUT}s")
    print(f"  Timing: ~{EXPECTED_S}s/exp expected | {c(YLW,f'>{WARN_S}s=slow')} | {c(RED,f'>{OVER_S}s=overtime')}")

    if args.dry_run:
        run_dry(exps)
        return

    if args.verify:
        run_verify(args.queue_file)
        return

    # ── Select experiments to run ─────────────────────────────────────────
    if args.only:
        run_list = [e for e in exps if e["name"] == args.only]
        if not run_list:
            print(c(RED, f"  ERROR: experiment '{args.only}' not found in queue"))
            sys.exit(1)
        print(f"\n  Running single experiment: {args.only}")
    elif args.test:
        run_list = exps[:1]
        print(f"\n{c(YLW, '  TEST MODE')} — running first experiment only")
        print(f"  After this, verify timing and run full queue if OK.")
    elif args.start > 1:
        run_list = exps[args.start - 1:]
        print(f"\n  Resuming from experiment #{args.start}")
    else:
        run_list = exps

    print()

    # ── Run loop ──────────────────────────────────────────────────────────
    results = []
    run_n = 0
    t_start = time.time()

    for exp in run_list:
        actual_idx = exps.index(exp) + 1

        if is_done(exp["name"]):
            print(f"[{actual_idx:02d}/{total}] {c(DIM, exp['name'] + ' — done, skipping')}")
            continue

        run_n += 1
        result = run_experiment(
            exp,
            f"logs/{exp['name']}.txt",
            actual_idx,
            total,
        )
        results.append(result)

        # Periodic summary
        if run_n % args.summary_every == 0:
            remaining = len([e for e in run_list[run_list.index(exp)+1:]
                             if not is_done(e["name"])])
            print_summary(results, prefix=f"After {run_n} — ")
            if remaining > 0:
                eta_s = remaining * (sum(r["elapsed"] for r in results) / len(results))
                print(f"  ETA: ~{eta_s/60:.0f}min remaining  "
                      f"({remaining} experiments left)\n")

    # ── Final output ──────────────────────────────────────────────────────
    total_min = (time.time() - t_start) / 60
    print(c(BOLD, "\n╔══════════════════════════════════════════════════════════════╗"))
    print(c(BOLD,  "║  COMPLETE                                                    ║"))
    print(c(BOLD,  "╚══════════════════════════════════════════════════════════════╝"))
    print(f"  Total wall time: {total_min:.1f}min")
    print_summary(results, prefix="Final ")
    print_top_n(results, n=10)

    # ── Problems report ───────────────────────────────────────────────────
    problems = [r for r in results if r["val_bpb"] is None or r["timed_out"] or r["exit_code"] != 0]
    if problems:
        print(f"\n  {c(RED, 'PROBLEMS — needs attention:')}")
        for r in problems:
            tag = "TIMEOUT" if r["timed_out"] else (f"exit={r['exit_code']}" if r["exit_code"] else "no_val_bpb")
            err = f" ({r['failure']})" if r.get("failure") else ""
            print(f"    {c(RED, '●')} {r['name']}  [{tag}]{err}")
    else:
        print(c(GRN, "\n  No problems — all experiments reported val_bpb ✓"))


if __name__ == "__main__":
    main()
