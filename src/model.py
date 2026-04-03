import torch
import torch.nn as nn

class PrefetchLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PrefetchLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Word embedding layer mapping integer IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Fully connected layer to map back to vocabulary space
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        # x is of shape (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # lstm_out contains all hidden states, we only need the final one
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # Taking output for the latest timestep
        last_timestep_out = lstm_out[:, -1, :] # (batch_size, hidden_dim)
        
        # Mapping to vocabulary dimension
        out = self.fc(last_timestep_out) # (batch_size, vocab_size)
        
        # Note: In PyTorch, CrossEntropyLoss expects unnormalized logits,
        # so we don't apply Softmax here during training. Softmax is only needed during inference.
        return out
