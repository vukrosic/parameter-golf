import subprocess
import os
import time
import sys

# Define experiments to run: (name, steps, env_overrides)
EXPERIMENTS = [
    ("arch_L10_D448_H8_KV8_M2_1000", 1000, {"NUM_LAYERS": "10", "MODEL_DIM": "448", "NUM_HEADS": "8", "NUM_KV_HEADS": "8", "MLP_MULT": "2", "MATRIX_LR": "0.06"}),
    ("arch_L9_D480_H8_KV8_M2_1000", 1000, {"NUM_LAYERS": "9", "MODEL_DIM": "480", "NUM_HEADS": "8", "NUM_KV_HEADS": "8", "MLP_MULT": "2", "MATRIX_LR": "0.06"}),
    ("arch_L12_D448_H8_KV8_M1_1000", 1000, {"NUM_LAYERS": "12", "MODEL_DIM": "448", "NUM_HEADS": "8", "NUM_KV_HEADS": "8", "MLP_MULT": "1", "MATRIX_LR": "0.06"}),
    ("arch_L15_D384_H6_KV6_M2_1000", 1000, {"NUM_LAYERS": "15", "MODEL_DIM": "384", "NUM_HEADS": "6", "NUM_KV_HEADS": "6", "MLP_MULT": "2", "MATRIX_LR": "0.06"}),
    ("arch_L20_D320_H5_KV5_M2_1000", 1000, {"NUM_LAYERS": "20", "MODEL_DIM": "320", "NUM_HEADS": "5", "NUM_KV_HEADS": "5", "MLP_MULT": "2", "MATRIX_LR": "0.06"}),
]

def get_gpu_status():
    """Returns a list of GPU utilization percentages from nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        return [int(x) for x in result.stdout.strip().split('\n')]
    except Exception as e:
        print(f"Error getting GPU status: {e}")
        return []

def run_experiment(gpu_id, name, steps, env):
    """Starts a subprocess to run the experiment on a specific GPU"""
    cmd_env = os.environ.copy()
    cmd_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd_env.update(env)
    
    # We use run_experiment.sh which handles logging and results dir
    log_file = f"logs/{name}_parallel.log"
    print(f"🚀 Starting {name} on GPU {gpu_id}. Logging to {log_file}")
    
    with open(log_file, "w") as f:
        return subprocess.Popen(
            ["bash", "lab/run_experiment.sh", name, str(steps)],
            env=cmd_env,
            stdout=f, stderr=subprocess.STDOUT,
            cwd=os.getcwd()
        )

def main():
    gpu_count = len(get_gpu_status())
    if gpu_count == 0:
        print("❌ No GPUs found.")
        sys.exit(1)
    
    print(f"🔍 Found {gpu_count} GPUs.")
    
    queue = list(EXPERIMENTS)
    running_processes = {} # gpu_id -> Popen object

    while queue or running_processes:
        # Check running processes
        finished_gpus = []
        for gpu_id, proc in running_processes.items():
            if proc.poll() is not None:
                print(f"✅ Experiment finished on GPU {gpu_id}")
                finished_gpus.append(gpu_id)
        
        for gpu_id in finished_gpus:
            del running_processes[gpu_id]
        
        # Try to assign new experiments
        if queue:
            for gpu_id in range(gpu_count):
                if gpu_id not in running_processes:
                    name, steps, env = queue.pop(0)
                    proc = run_experiment(gpu_id, name, steps, env)
                    running_processes[gpu_id] = proc
                    if not queue:
                        break
        
        time.sleep(10)

if __name__ == "__main__":
    main()
