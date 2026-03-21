---
name: kill-exp
description: Stop a running experiment on a specific GPU and free it for new work. Use when the user wants to cancel, stop, abort, or kill an experiment, or free up a GPU.
---

# Kill Experiment

Stop a running experiment on a GPU instance.

## Instructions

1. Source credentials from `lab/gpu_creds.sh` in `/root/parameter-golf/`.

2. If the user specifies an experiment name, find which GPU it's running on:
   - SSH into each GPU and check `ps aux | grep train_gpt`
   - Match the experiment name from the process arguments or log file names

3. If the user specifies a GPU name (e.g., "L40S", "5090"), target that GPU directly.

4. **Before killing**, show the user:
   - What experiment is running
   - Current step progress (from latest log line)
   - How long it's been running
   - Ask for confirmation

5. Kill the experiment:
   ```bash
   # Kill the training process and any child processes
   pkill -f "train_gpt.py"
   # Or more targeted:
   kill <PID>
   ```

6. Also kill the queue runner if one is active:
   ```bash
   pkill -f "run_queue.sh"
   pkill -f "run_experiment.sh"
   ```

7. Verify the process is stopped: `ps aux | grep train_gpt | grep -v grep`

8. Report back: experiment stopped, GPU is now free.

## SSH Pattern
```bash
sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no -p $PORT root@$HOST "cd /root/parameter-golf && <cmd>"
```

## Important
- Always confirm with the user before killing — they may have been running something important
- If partial results exist, suggest collecting them first with `/collect`
- The killed experiment can be re-run later since `run_queue.sh` skips completed experiments (those with `results/<name>/train.log`)
