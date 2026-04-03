import os
import shutil
import subprocess
import yaml
import json
import time

ITERATIONS = 1

apps = [
    #{"app_name": "chrome", "target_app": "google-chrome"},
    #{"app_name": "gimp", "target_app": "gimp"},
     {"app_name": "firefox", "target_app": "firefox"}
]

# Hyperparameter strategies for improvement across exactly 10 iterations
hyperparams = [
    {"seq_length": 40, "embedding_dim": 64, "hidden_dim": 128, "learning_rate": 0.001, "epochs": 30},
    {"seq_length": 40, "embedding_dim": 64, "hidden_dim": 128, "learning_rate": 0.001, "epochs": 40},
    {"seq_length": 50, "embedding_dim": 128, "hidden_dim": 256, "learning_rate": 0.0005, "epochs": 40},
    {"seq_length": 60, "embedding_dim": 128, "hidden_dim": 256, "learning_rate": 0.0005, "epochs": 50},
    {"seq_length": 60, "embedding_dim": 256, "hidden_dim": 256, "learning_rate": 0.0001, "epochs": 60},
    {"seq_length": 60, "embedding_dim": 256, "hidden_dim": 512, "learning_rate": 0.0001, "epochs": 60},
    {"seq_length": 70, "embedding_dim": 256, "hidden_dim": 512, "learning_rate": 0.0001, "epochs": 70},
    {"seq_length": 70, "embedding_dim": 512, "hidden_dim": 512, "learning_rate": 0.00005, "epochs": 70},
    {"seq_length": 80, "embedding_dim": 512, "hidden_dim": 512, "learning_rate": 0.00005, "epochs": 80},
    {"seq_length": 80, "embedding_dim": 512, "hidden_dim": 1024, "learning_rate": 0.00001, "epochs": 80},
]

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_config(config, path="config/config.yaml"):
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def run_pipeline():
    python_exec = "./venv/bin/python" if os.path.exists("./venv/bin/python") else "python3"
    
    stages = ['collect', 'process', 'train', 'evaluate']
    for stage in stages:
        print(f"Running {stage}...")
        res = subprocess.run([python_exec, "main.py", stage])
        if res.returncode != 0:
            print(f"Error during {stage}")
            return False
    return True

def copy_results(app_name, iteration, run_folder_name):
    dest_dir = f"data/results/{run_folder_name}/{iteration}_{app_name}"
    os.makedirs(dest_dir, exist_ok=True)
    
    results_dir = "data/results"
    for filename in os.listdir(results_dir):
        file_path = os.path.join(results_dir, filename)
        if os.path.isfile(file_path):
            shutil.copy2(file_path, os.path.join(dest_dir, filename))
            print(f"Copied {filename} to {dest_dir}")

def main():
    print("Starting Multi-App Optimization Orchestrator")
    config = load_config()
    
    # Generate unique folder name for this entire run (the 'set of iterations')
    from datetime import datetime
    run_folder_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"All iteration results for this execution will be saved in: data/results/{run_folder_name}/")
    
    for app in apps:
        app_name = app["app_name"]
        target_app = app["target_app"]
        print(f"\n==========================================")
        print(f"Optimizing for {app_name} ({target_app})")
        print(f"==========================================")
        
        for i in range(ITERATIONS):
            iteration = i + 1
            print(f"\n--- Iteration {iteration}/{ITERATIONS} for {app_name} ---")
            
            # Apply hyperparameters
            hp = hyperparams[i]
            config["model"]["seq_length"] = hp["seq_length"]
            config["model"]["embedding_dim"] = hp["embedding_dim"]
            config["model"]["hidden_dim"] = hp["hidden_dim"]
            config["model"]["learning_rate"] = hp["learning_rate"]
            config["model"]["epochs"] = hp["epochs"]
            
            config["system"]["app_name"] = app_name
            config["system"]["target_app"] = target_app
            
            save_config(config)
            
            # Run pipeline
            success = run_pipeline()
            if not success:
                print(f"Iteration {iteration} for {app_name} failed. Skipping remaining stages.")
            
            # Save results into iteration folder
            copy_results(app_name, iteration, run_folder_name)
            
            # Give a brief pause
            time.sleep(2)
            
if __name__ == "__main__":
    main()
