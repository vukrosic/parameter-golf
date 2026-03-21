# Remote GPU Control — Lessons Learned

Only care about this on the main instance where you are running, these are rules for you, the AI agent.

## Connection Details

### GPU 1: RTX 3090 (remote)
- **Host**: `proxy.us-ca-6.gpu-instance.novita.ai`
- **Port**: `62248`
- **User**: `root`
- **Password**: `QVYvzTEyhLequoOMQEHJ`
- **GPU**: NVIDIA RTX 3090 (24GB VRAM), ~2,702ms/step
- **Helper**: `/tmp/remote.sh` (if created)

### GPU 2: L40S
- **Host**: `proxy.us-ca-6.gpu-instance.novita.ai`
- **Port**: `62396`
- **User**: `root`
- **Password**: `ArtbZJu7nmE4VSwgBCQl`
- **GPU**: NVIDIA L40S (48GB VRAM), ~950ms/step (torch.compile)
- **Helper**: `/tmp/remote3.sh`
- **Queue**: `lab/queue_gpu3.txt`

### GPU 3: RTX 5090
- **Host**: `proxy.us-ca-6.gpu-instance.novita.ai`
- **Port**: `62132`
- **User**: `root`
- **Password**: `qxCByieIPNPkHGt9NaWF`
- **GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM), ~605ms/step (torch.compile)
- **Helper**: `/tmp/remote4.sh`
- **Queue**: `lab/queue_gpu4.txt`

## Critical Mistakes & Fixes

### 1. SSH commands default to /root/, NOT the repo directory

Every SSH command starts in `/root/`. The repo is cloned to `/root/parameter-golf/`.
You MUST prefix every remote command with `cd /root/parameter-golf && ...`.

**Wrong:**
```bash
sshpass -p 'PASS' ssh ... 'git checkout lab'           # runs in /root/, fails
sshpass -p 'PASS' ssh ... 'python3 train_gpt.py'       # file not found
```

**Right:**
```bash
sshpass -p 'PASS' ssh ... 'cd /root/parameter-golf && git checkout lab'
sshpass -p 'PASS' ssh ... 'cd /root/parameter-golf && python3 train_gpt.py'
```

### 2. Use git -C for git commands (avoids cd)

```bash
sshpass -p 'PASS' ssh ... 'git -C /root/parameter-golf checkout lab'
sshpass -p 'PASS' ssh ... 'git -C /root/parameter-golf pull origin lab'
```

### 3. Helper script pattern

Create `/tmp/remote.sh` for convenience, but the script itself does NOT cd into the repo.
The caller must always include the cd:

```bash
/tmp/remote.sh 'cd /root/parameter-golf && python3 train_gpt.py'
```

### 4. Long-running training over SSH

For long experiments, use `nohup` to survive SSH disconnects:

```bash
/tmp/remote.sh 'cd /root/parameter-golf && nohup bash -c "RUN_ID=exp1 ITERATIONS=500 ... python3 train_gpt.py" > /root/parameter-golf/logs/exp1.txt 2>&1 &'
```

Then check status:
```bash
/tmp/remote.sh 'nvidia-smi | grep python'       # GPU occupied?
/tmp/remote.sh 'tail -5 /root/parameter-golf/logs/exp1.txt'  # progress?
```

### 5. GitHub sync workflow

**After experiment finishes on remote:**
```bash
# On remote: commit results and push
/tmp/remote.sh 'cd /root/parameter-golf && git add results/ && git commit -m "results: ..." && git push origin lab'

# On local: pull
git pull origin lab
```

**Before starting remote experiment: sync code changes**
```bash
# On local: push any code changes
git add . && git commit -m "..." && git push origin lab

# On remote: pull
/tmp/remote.sh 'cd /root/parameter-golf && git pull origin lab'
```

## Step Timing

| GPU | Step avg (ms) | 500 steps | 4000 steps | 13780 steps |
|-----|--------------|-----------|------------|-------------|
| RTX 5090 (remote) | ~605 | ~5 min | ~40 min | ~2.3 hrs |
| L40S (remote) | ~950 | ~8 min | ~63 min | ~3.6 hrs |
| RTX 3090 (local) | ~2629 | ~22 min | ~175 min | ~10.1 hrs |
| RTX 3090 (remote) | ~2702 | ~23 min | ~180 min | ~10.3 hrs |

The wallclock budget in run_experiment.sh is calibrated for L40S (3.4s/step * 1.15 buffer).
On faster GPUs this gives massive extra buffer, which is fine (step-based termination dominates).

## Environment Setup (done)

All 4 machines have:
- Python 3.11.13
- torch 2.10.0+cu128
- All deps from requirements.txt
- Data: fineweb10B_sp1024 (1 shard + val)
- Branch: lab

GPU capabilities:
- RTX 3090: SM 8.6 (eager mode, no torch.compile)
- L40S: SM 8.9 (torch.compile enabled)
- RTX 5090: SM 12.0 (torch.compile enabled, Blackwell)
