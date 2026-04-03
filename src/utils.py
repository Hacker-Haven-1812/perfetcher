import yaml
import json
import os

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
