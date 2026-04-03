import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_effectiveness(metrics_data, out_dir):
    eff = metrics_data.get('prefetch_effectiveness', {})
    if not eff: return
    
    keys = ['coverage', 'hit_rate', 'miss_rate', 'pollution_rate']
    values = [eff.get(k, 0) for k in keys]
    
    plt.figure(figsize=(8,5))
    sns.barplot(x=keys, y=values, palette='magma')
    plt.ylim(0, 1.0)
    plt.title('Prefetch Cache Effectiveness')
    
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
        
    out_path = os.path.join(out_dir, 'prefetch_effectiveness.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

def plot_speedup(metrics_data, out_dir):
    perf = metrics_data.get('performance', {})
    if not perf: return
    
    cold_time = perf.get('cold_time_median_sec', 0)
    pref_time = perf.get('prefetched_time_median_sec', 0)
    
    labels = ['Cold Start', 'Prefetched Start']
    times = [cold_time, pref_time]
    
    plt.figure(figsize=(6,5))
    sns.barplot(x=labels, y=times, palette=['#e74c3c', '#2ecc71'])
    plt.ylabel('Startup True I/O Time (Seconds)')
    plt.title('Application Launch Speedup Comparison')
    
    for i, v in enumerate(times):
        plt.text(i, v + max(times)*0.02 if max(times) > 0 else 0, f"{v:.3f}s", ha='center')
        
    out_path = os.path.join(out_dir, 'speedup_comparison.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 plot_all_metrics.py <metrics_json_file>")
        sys.exit(1)
        
    metrics_file = sys.argv[1]
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            data = json.load(f)
            
        out_dir = os.path.dirname(metrics_file)
        plot_effectiveness(data, out_dir)
        plot_speedup(data, out_dir)
