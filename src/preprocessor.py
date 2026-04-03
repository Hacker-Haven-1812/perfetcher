import os
import torch
from collections import Counter
from src.utils import load_config, ensure_dir, save_json

def preprocess_logs():
    config = load_config()
    app_name = config['system']['app_name']
    seq_length = config['model']['seq_length']
    raw_path = config['data']['raw_path']
    processed_path = config['data']['processed_path']
    ensure_dir(processed_path)
    
    log_file = os.path.join(raw_path, f"{app_name}_log.txt")
    vocab_file = os.path.join(processed_path, f"{app_name}_vocab.json")
    data_file = os.path.join(processed_path, f"{app_name}_data.pt")
    
    if not os.path.exists(log_file):
        print(f"Error: Log file {log_file} not found. Run 'collect' first.")
        return
        
    with open(log_file, "r") as f:
        file_sequence = [line.strip() for line in f if line.strip()]
        
    print(f"Loaded {len(file_sequence)} file accesses. Building vocabulary...")
    

    
    # Reserved tokens
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for filepath in file_sequence:
        if filepath not in vocab:
            vocab[filepath] = idx
            idx += 1
            
    # Save vocabulary
    save_json(vocab, vocab_file)
    print(f"Vocabulary built with {len(vocab)} unique files. Saved to {vocab_file}")
    
    # Conditionally build static frequency profile for hybrid preloading ONLY if app is heavily reliant on dependencies
    static_file = os.path.join(processed_path, f"{app_name}_static_freq.json")
    if len(vocab) > 1500:
        freq_counter = Counter(file_sequence)
        top_files = [f for f, count in freq_counter.most_common(1500)]
        save_json(top_files, static_file)
        print(f"Heavy Application detected. Saved {len(top_files)} highly frequent files to static profile for hybrid prefetching.")
    else:
        if os.path.exists(static_file):
            os.remove(static_file)
        print(f"Lightweight Application detected. Relying purely on dynamic LSTM predictive caching.")
    
    # Convert string sequence to integer sequence
    int_sequence = [vocab.get(f, 1) for f in file_sequence]
    
    # We will use a sliding window logic to generate matching input-output pairs
    # Input is sequence of file IDs (length: seq_length), Output is next file ID
    X, Y = [], []
    for i in range(len(int_sequence) - seq_length):
        X.append(int_sequence[i : i + seq_length])
        Y.append(int_sequence[i + seq_length])
        
    if not X:
        print("Error: The collected logs sequence is shorter than 'seq_length'.")
        print("Try collecting logs again or reducing seq_length in config.yaml.")
        return
        
    X_tensor = torch.tensor(X, dtype=torch.long)
    Y_tensor = torch.tensor(Y, dtype=torch.long)
    
    # Save tensor data
    torch.save({'X': X_tensor, 'Y': Y_tensor}, data_file)
    print(f"Generated {len(X)} training samples. Saved dataset to {data_file}")

if __name__ == "__main__":
    preprocess_logs()
