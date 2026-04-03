import os
import torch
from src.model import PrefetchLSTM
from src.utils import load_config, load_json

def do_prefetch(input_sequence_filepaths=None, top_k=5, max_predictions=300):
    """
    Predicts files likely to be needed soon and loads them into memory.
    """
    config = load_config()
    app_name = config['system']['app_name']
    embedding_dim = config['model']['embedding_dim']
    hidden_dim = config['model']['hidden_dim']
    seq_length = config['model']['seq_length']
    
    processed_path = config['data']['processed_path']
    models_path = config['data']['models_path']
    vocab_file = os.path.join(processed_path, f"{app_name}_vocab.json")
    model_file = os.path.join(models_path, f"{app_name}_model.pt")
    static_file = os.path.join(processed_path, f"{app_name}_static_freq.json")
    raw_log = os.path.join(config['data']['raw_path'], f"{app_name}_log.txt")
    
    if not os.path.exists(model_file) or not os.path.exists(vocab_file):
        print("Error: Model or vocabulary not found. Train the model first.")
        return []

    vocab = load_json(vocab_file)
    vocab_size = len(vocab)
    idx_to_file = {v: k for k, v in vocab.items()}
    
    model = PrefetchLSTM(vocab_size, embedding_dim, hidden_dim)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    
    # Use the beginning of recorded log as a "seed" sequence to initiate prediction
    if not input_sequence_filepaths:
        if os.path.exists(raw_log):
            with open(raw_log, "r") as f:
                lines = f.read().splitlines()
                input_sequence_filepaths = lines[:seq_length] if len(lines) >= seq_length else lines
        else:
            print("Warning: No raw log found for seed. Using empty start.")
            input_sequence_filepaths = []
            
    # Pad if seed is shorter than seq_length
    if len(input_sequence_filepaths) < seq_length:
        pad_len = seq_length - len(input_sequence_filepaths)
        input_sequence_filepaths = ["<PAD>"] * pad_len + input_sequence_filepaths
        
    int_sequence = [vocab.get(f, 1) for f in input_sequence_filepaths[-seq_length:]]
    current_input = torch.tensor([int_sequence], dtype=torch.long)
    
    prefetched_files = set()
    total_latency_ms = 0
    
    # Hybrid Step 1: Sweep bulk-load static dependencies dynamically pulled from historical distribution
    if os.path.exists(static_file):
        top_static = load_json(static_file)
        for filepath in top_static:
            if filepath not in ["<PAD>", "<UNK>"]:
                prefetched_files.add(filepath)
        print(f"Loaded {len(top_static)} highly frequent files instantly from historical profile.")
    
    # Hybrid Step 2: Use Autoregressive LSTM for subsequent chronological loading
    import time
    with torch.no_grad():
        for _ in range(max_predictions):
            start_infer = time.perf_counter()
            output = model(current_input)
            probs = torch.softmax(output, dim=-1)[0]
            
            top_k_indices = torch.topk(probs, k=min(top_k, vocab_size)).indices.tolist()
            infer_ms = (time.perf_counter() - start_infer) * 1000
            total_latency_ms += infer_ms
            
            for idx in top_k_indices:
                if probs[idx] > 0.05: # Probability threshold avoids polluting cache with low confidence guesses
                    filepath = idx_to_file.get(idx)
                    if filepath and filepath not in ["<PAD>", "<UNK>"]:
                        prefetched_files.add(filepath)
            
            # Take the top prediction for the autoregressive step
            next_idx = top_k_indices[0]
            if probs[next_idx] < 0.01:
                # Stop if our confidence falls very low to avoid catastrophic error accumulation
                break
                
            current_input = torch.cat([current_input[:, 1:], torch.tensor([[next_idx]])], dim=1)
            
    # The actual OS-level prefetch
    successful_prefetches = execute_prefetch(list(prefetched_files))
    return successful_prefetches

def execute_prefetch(file_paths):
    prefetched = []
    print(f"Predicting and loading {len(file_paths)} distinct files into page cache...")
    for f in file_paths:
        try:
            if os.path.isfile(f):
                # Read up to 4MB of the file to force it into OS page cache
                with open(f, 'rb') as fp:
                    fp.read(4 * 1024 * 1024) 
                prefetched.append(f)
        except Exception:
            # File might have restricted permissions or no longer exist
            pass
            
    print(f"Successfully loaded {len(prefetched)} files into cache.")
    return prefetched

if __name__ == "__main__":
    do_prefetch()
