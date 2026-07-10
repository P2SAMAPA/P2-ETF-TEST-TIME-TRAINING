import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TTTNetwork(nn.Module):
    """
    Network with Test-Time Training Layers.
    The last layer continues to update via self-supervised loss during inference.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Pretrained layers (frozen during test-time)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Test-time trainable layers
        self.ttt_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.ttt_fc2 = nn.Linear(hidden_size // 2, 1)
        # Auxiliary decoder for self-supervised loss (reconstruction)
        self.decoder = nn.Linear(hidden_size // 2, input_size)  # reconstruct full input
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Use last hidden state
        h = lstm_out[:, -1, :]
        # Test-time trainable layers
        h = self.relu(self.ttt_fc1(h))
        out = self.ttt_fc2(h)
        return out.squeeze(-1)

    def forward_with_aux(self, x):
        """Forward pass with auxiliary reconstruction output for self-supervised loss."""
        lstm_out, _ = self.lstm(x)
        h = lstm_out[:, -1, :]
        h = self.relu(self.ttt_fc1(h))
        out = self.ttt_fc2(h)
        # Auxiliary reconstruction: reconstruct the entire input sequence
        reconstructed = self.decoder(h)  # (batch, input_size)
        return out.squeeze(-1), reconstructed

    def adapt(self, x, lr=0.0001, steps=5):
        """
        Adapt the test-time trainable layers on a new sample using self-supervised loss.
        """
        self.train()
        # Create optimizer for TTT layers only
        ttt_params = list(self.ttt_fc1.parameters()) + list(self.ttt_fc2.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(ttt_params, lr=lr)
        for _ in range(steps):
            # Forward pass
            _, reconstructed = self.forward_with_aux(x)
            # Self-supervised loss: reconstruct the last time step of the input
            target = x[:, -1, :]  # (batch, input_size)
            # Now both have size (batch, input_size)
            loss = F.mse_loss(reconstructed, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def prepare_data(returns, macro_df, seq_len=10):
    """
    Prepare sequences for training.
    returns: pandas Series (single ETF)
    macro_df: pandas DataFrame (macro variables)
    """
    if len(returns) < seq_len + 1:
        return None, None
    common_idx = returns.index.intersection(macro_df.index)
    ret_aligned = returns.loc[common_idx]
    macro_aligned = macro_df.loc[common_idx]
    X, y = [], []
    for i in range(seq_len, len(ret_aligned)):
        ret_seq = ret_aligned.iloc[i-seq_len:i].values.reshape(-1, 1)
        macro_seq = macro_aligned.iloc[i-seq_len:i].values
        seq_features = np.concatenate([ret_seq, macro_seq], axis=1)
        X.append(seq_features)
        y.append(ret_aligned.iloc[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

def ttt_score(returns, macro_df, hidden_size=64, num_layers=2, seq_len=10, epochs=20, lr=0.001, batch_size=16, ttt_lr=0.0001, ttt_steps=5):
    """
    Train TTT network and return predicted next-day return with momentum enhancement.
    """
    X, y = prepare_data(returns, macro_df, seq_len)
    if X is None or len(X) < batch_size:
        return 0.0
    input_size = X.shape[2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TTTNetwork(input_size, hidden_size, num_layers).to(device)
    # Pretraining
    dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    # Test-time adaptation on the most recent sequence
    model.eval()
    # Use the last sequence for adaptation
    ret_seq = returns.iloc[-seq_len:].values.reshape(-1, 1)
    macro_seq = macro_df.iloc[-seq_len:].values
    last_seq = np.concatenate([ret_seq, macro_seq], axis=1)
    last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
    # Adapt the model on the new sample
    model.adapt(last_seq_tensor, lr=ttt_lr, steps=ttt_steps)
    # Predict
    model.eval()
    with torch.no_grad():
        pred = model(last_seq_tensor).item()
    # Momentum factor
    last_return = returns.iloc[-1]
    momentum = 1.0 + last_return
    momentum = max(0.5, min(2.0, momentum))
    final_score = pred * momentum
    return float(final_score)
