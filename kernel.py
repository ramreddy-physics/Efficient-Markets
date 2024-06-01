import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def split_data(X, Y, tickers):
    (
        X_train,
        X_temp,
        Y_train,
        Y_temp,
        tickers_train,
        tickers_temp,
    ) = train_test_split(X, Y, tickers, test_size=0.1, random_state=42, shuffle=True)
    X_val, X_test, Y_val, Y_test, tickers_val, tickers_test = train_test_split(
        X_temp, Y_temp, tickers_temp, test_size=0.7, random_state=42, shuffle=True
    )
    return (
        X_train,
        Y_train,
        tickers_train,
        X_val,
        Y_val,
        tickers_val,
        X_test,
        Y_test,
        tickers_test,
    )

def scoring_accuracy(outputs, labels):
    return (np.sign(outputs)==np.sign(labels)).mean()

def process_input_RNN(X):
    X_rnn = []
    for x in X:
        a1 = x[:120]
        a2 = x[120:240]
        a3 = x[240:360]
        a4 = x[360:]
        a = np.vstack((a1, a2, a3, a4)).transpose(1, 0)
        X_rnn.append(a)
    X_rnn = np.array(X_rnn)
    return X_rnn


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(X.device)

        # RNN for X1
        out1, _ = self.rnn(X, h0)
        out1 = out1[:, -1, :]  # take the last output from the sequence

        # Pass through fully connected layers
        out = self.fc1(out1)
        return out


def train(X, Y, tickers):
    (
        X_train,
        Y_train,
        tickers_train,
        X_val,
        Y_val,
        tickers_val,
        X_test,
        Y_test,
        tickers_test,
    ) = split_data(X, Y, tickers)
    X_train_rnn = process_input_RNN(X_train)
    X_val_rnn = process_input_RNN(X_val)
    X_test_rnn = process_input_RNN(X_test)

    X_train_tensor = torch.tensor(X_train_rnn, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_rnn, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val_rnn, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).view(-1, 1)

    """
    Model Architecture
    """
    input_size = 4
    hidden_size = 32
    output_size = 1
    num_layers = 4

    model_rnn = RNNModel(input_size, hidden_size, output_size, num_layers)

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    criterion = nn.SmoothL1Loss()

    optimizer = optim.Adam(model_rnn.parameters(), lr=1.0e-04)

    # Training settings
    num_epochs = 1000
    patience = 25
    max_grad_norm = 1.0

    # Lists to store losses
    train_losses = []
    val_losses = []
    test_losses = []

    best_val_loss = float("inf")
    best_model_weights = model_rnn.state_dict()
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        model_rnn.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model_rnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_rnn.parameters(), max_grad_norm)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model_rnn.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model_rnn(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model_rnn.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                model_rnn.load_state_dict(best_model_weights)
                patience_counter = 0
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.1
                patience_counter = 0

    model_rnn.load_state_dict(best_model_weights)

    Y_pred_test = model_rnn(X_test_tensor).detach().numpy().flatten()
    torch.save(model_rnn.state_dict(), 'best_model.pth')

    return scoring_accuracy(Y_pred_test, Y_test)
