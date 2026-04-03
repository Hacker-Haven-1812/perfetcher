import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from src.model import PrefetchLSTM
from src.utils import load_config, ensure_dir, load_json

def train_model():
    config = load_config()
    app_name = config['system']['app_name']
    learning_rate = config['model']['learning_rate']
    epochs = config['model']['epochs']
    embedding_dim = config['model']['embedding_dim']
    hidden_dim = config['model']['hidden_dim']
    
    processed_path = config['data']['processed_path']
    models_path = config['data']['models_path']
    ensure_dir(models_path)
    
    data_file = os.path.join(processed_path, f"{app_name}_data.pt")
    vocab_file = os.path.join(processed_path, f"{app_name}_vocab.json")
    model_file = os.path.join(models_path, f"{app_name}_model.pt")
    
    if not os.path.exists(data_file) or not os.path.exists(vocab_file):
        print("Error: Processed data or vocabulary missing. Run 'process' first.")
        return
        
    print("Loading data and vocabulary...")
    data = torch.load(data_file)
    X, Y = data['X'], data['Y']
    
    vocab = load_json(vocab_file)
    vocab_size = len(vocab)
    
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = PrefetchLSTM(vocab_size, embedding_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Starting training for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), model_file)
    print(f"Training completed. Model saved to {model_file}")

if __name__ == "__main__":
    train_model()
