import os
import subprocess
import time
import statistics
import json
import sys
from src.utils import load_config, save_json, ensure_dir
from src.metrics import MetricsCalculator
from src.prefetcher import do_prefetch

def drop_caches():
    print("Dropping OS page caches. (Requires sudo)")
    try:
        subprocess.run(["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"], check=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("Warning: Could not drop caches. Results may not reflect true cold start.")

def measure_app_launch(config):
    target_app = config['system']['target_app']
    app_name = config['system']['app_name']
    
    # Trace internal system call metric (time spent blocked) using -T flag
    tmp_trace = f"/tmp/eval_strace_{app_name}.txt"
    strace_cmd = ["strace", "-e", "trace=open,openat", "-T", "-o", tmp_trace, target_app]
    
    start_time = time.perf_counter()
    try:
        process = subprocess.Popen(strace_cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        time.sleep(10) # Let the application start up fully (extended for large datasets)
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
    except FileNotFoundError:
        print(f"Error: Could not launch {target_app}")
        return 0.0, []
        
    end_time = time.perf_counter()
        
    files_opened = []
    total_io_wait_time = 0.0
    if os.path.exists(tmp_trace):
        with open(tmp_trace, 'r') as f:
            for line in f:
                if 'openat(' in line or 'open(' in line:
                    # Parse true IO time from trailing component e.g. = 3 <0.000123>
                    try:
                        if '<' in line and '>' in line:
                            time_str = line.split('<')[-1].split('>')[0]
                            io_time = float(time_str)
                            total_io_wait_time += io_time
                    except ValueError:
                        pass
                        
                    file_parts = line.split('"')
                    if len(file_parts) >= 3:
                        filepath = file_parts[1]
                        if filepath.startswith("/") and not filepath.startswith("/dev"):
                            if os.path.exists(filepath):
                                files_opened.append(filepath)
                            
    # The true IO latency metric
    launch_time = total_io_wait_time if total_io_wait_time > 0 else (end_time - start_time)
    return launch_time, files_opened

def evaluate_system():
    config = load_config()
    results_path = config['data']['results_path']
    ensure_dir(results_path)
    
    print("\n=== COLD START EVALUATION ===")
    cold_times = []
    cold_accesses = set()
    for i in range(2):
        drop_caches()
        print(f"Running Cold Start {i+1}...")
        t, files = measure_app_launch(config)
        cold_times.append(t)
        cold_accesses.update(files)
        
    cold_time_median = statistics.median(cold_times) if cold_times else 0.0
    
    print("\n=== PREFETCHED EVALUATION ===")
    prefetched_times = []
    prefetched_accesses = set()
    last_prefetched_files = []
    
    for i in range(2):
        drop_caches()
        print(f"Prefetching files into cache {i+1}...")
        last_prefetched_files = do_prefetch()
        
        print(f"Running Prefetched Start {i+1}...")
        t, files = measure_app_launch(config)
        prefetched_times.append(t)
        prefetched_accesses.update(files)
        
    pref_time_median = statistics.median(prefetched_times) if prefetched_times else 0.0
    
    speedup = ((cold_time_median - pref_time_median) / cold_time_median) * 100 if cold_time_median > 0 else 0
    print(f"\nCold Start Median IO Time: {cold_time_median:.4f}s")
    print(f"Prefetched Start Median IO Time: {pref_time_median:.4f}s")
    print(f"Speedup: {speedup:.2f}%\n")
    
    # Calculate all metrics
    metrics_calc = MetricsCalculator()
    metrics_calc.populate_from_evaluation(last_prefetched_files, cold_accesses)
    metrics_data = metrics_calc.compute_all_metrics()
    
    metrics_data["performance"]["speedup_percent"] = speedup
    metrics_data["performance"]["cold_time_median_sec"] = cold_time_median
    metrics_data["performance"]["prefetched_time_median_sec"] = pref_time_median
    
    metrics_file = os.path.join(results_path, "evaluation_metrics.json")
    save_json(metrics_data, metrics_file)
    print(f"Evaluation metrics saved to {metrics_file}")
    
    print("Generating graphs...")
    subprocess.run([sys.executable, "scripts/generate_graph.py", metrics_file])
    subprocess.run([sys.executable, "scripts/plot_all_metrics.py", metrics_file])
    print("Evaluation completed successfully.")

if __name__ == "__main__":
    evaluate_system()
