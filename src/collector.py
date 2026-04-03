import subprocess
import os
import time
from src.utils import load_config, ensure_dir

def collect_logs():
    config = load_config()
    target_app = config['system']['target_app']
    app_name = config['system']['app_name']
    raw_path = config['data']['raw_path']
    runs = config['system'].get('collect_runs', 1)
    ensure_dir(raw_path)
    
    log_file = os.path.join(raw_path, f"{app_name}_log.txt")
    tmp_trace = "/tmp/strace_output.txt"
    
    print(f"Starting {target_app} to collect file access traces over {runs} runs...")
    
    for run in range(runs):
        print(f"--- Collect Run {run+1}/{runs} ---")
        # Launch the application and trace `open` and `openat` system calls.
        strace_cmd = ["strace", "-e", "trace=open,openat", "-o", tmp_trace, target_app]
        
        try:
            process = subprocess.Popen(strace_cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            
            # Wait 8 seconds to capture the cold-start behavior
            time.sleep(8)
            
            # Terminate the application after capturing startup
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        except FileNotFoundError:
            print(f"Error: Could not run '{target_app}'. Is it installed?")
            return

        print("Parsing trace logs...")
        files_accessed = []
        if os.path.exists(tmp_trace):
            with open(tmp_trace, "r") as f:
                for line in f:
                    if 'openat(' in line or 'open(' in line:
                        # Parse the file path which is enclosed in double quotes
                        parts = line.split('"')
                        if len(parts) >= 3:
                            filepath = parts[1]
                            # Filter out virtual filesystems and non-absolute paths
                            if filepath.startswith("/") and not any(filepath.startswith(prefix) for prefix in ['/dev', '/proc', '/sys', '/tmp', '/run']):
                                if os.path.exists(filepath):
                                    files_accessed.append(filepath)
                                    
        with open(log_file, "a") as f:
            for filepath in files_accessed:
                f.write(filepath + "\n")
                
        print(f"Appended {len(files_accessed)} file accesses to {log_file}")
        time.sleep(2) # Give system a short break between runs

if __name__ == "__main__":
    collect_logs()
