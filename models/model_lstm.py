"""
This part was modified based on Willy's LSTM architecture.
Reference: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# LSTM architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=1024, output_size=7, num_layers=4, dropout=0.1,output_seq=20, device=None):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.output_seq = output_seq
        
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers=num_layers,batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size,output_size)

    def forward(self, input_seq, state):
        lstm_out, state = self.lstm(input_seq, state)  # lstm_out shape: (batch_size, 20, hidden_size)
        predictions = self.linear(lstm_out[:,-(self.output_seq):,:])

        return predictions, state

# Training epoch
def train_epoch(trainloader, lstm_model, optimizer, criterion, device):
    lstm_model.train()
    total_loss = 0

    for train_input, train_target, input_len, target_len in trainloader:
        train_input = train_input.to(device)
        train_target = train_target.to(device)
        
        batch_size = train_input.size(0)

        # 初始化隱藏狀態和細胞狀態
        # 依照育堂的建議，這個部分的初始化應該要在forward中定義
        h_0 = torch.zeros(lstm_model.num_layers, batch_size, lstm_model.hidden_size, device=device)
        c_0 = torch.zeros(lstm_model.num_layers, batch_size, lstm_model.hidden_size, device=device)
        state = (h_0,c_0)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        lstm_outputs, _ = lstm_model(train_input, state)


        # Backpropagation
        # print(lstm_outputs.shape,train_target.shape)
        loss = criterion(lstm_outputs, train_target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(trainloader)

# Evaluation epoch
def eval_epoch(valloader, lstm_model, criterion, device):
    lstm_model.eval()
    total_loss = 0
    preds = []
    targets= []

    with torch.no_grad():
        for val_input, val_target, input_len, target_len in valloader:
            val_input = val_input.to(device)
            val_target = val_target.to(device)

            batch_size = val_input.size(0)
            
            # 初始化隱藏狀態和細胞狀態
            h_0 = torch.zeros(lstm_model.num_layers, batch_size, lstm_model.hidden_size, device=device)
            c_0 = torch.zeros(lstm_model.num_layers, batch_size, lstm_model.hidden_size, device=device)
            state = (h_0,c_0)

            # Forward pass
            lstm_outputs, _ = lstm_model(val_input,state)

            # Loss
            loss = criterion(lstm_outputs, val_target)
            total_loss += loss.item()

            # log output
            pred = lstm_outputs.cpu().detach().numpy()
            target = val_target.cpu().detach().numpy()
            preds.append(pred)
            targets.append(target)
    
    angle_actual = []
    angle_pred = []

    for target, pred in zip(targets, preds):
        angle_actual.append(target[:, :, -1])  # shape: [batch_size, seq_len]
        angle_pred.append(pred[:, :, -1])      # shape: [batch_size, seq_len]

    angle_actual_flatten = np.concatenate([seq.flatten() for seq in angle_actual])
    angle_pred_flatten = np.concatenate([seq.flatten() for seq in angle_pred])

    mse = mean_squared_error(angle_actual_flatten, angle_pred_flatten)
    mae = mean_absolute_error(angle_actual_flatten, angle_pred_flatten)
    
    return total_loss / len(valloader), mse, mae

# The overall training procedure
def running_train(train_dataloader, val_dataloader, lstm_model, optimizer, 
                    criterion, n_epochs, best_model_path, device):
    
    input_size = lstm_model.input_size
    hidden_size = lstm_model.hidden_size
    output_size = lstm_model.output_size

    best_loss = float('inf')
    train_losses = []
    val_losses = []

    # dynamic disable tqdm output
    disable_tqdm = not sys.stdout.isatty()    
    
    for epoch in tqdm(range(1, n_epochs + 1), desc="Training Progress", disable=disable_tqdm):
        train_loss = train_epoch(train_dataloader, lstm_model, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        val_loss, mse, mae = eval_epoch(val_dataloader, lstm_model, criterion, device)
        val_losses.append(val_loss)

        #Save the model if the evaluation loss decreases
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch':epoch,
                'model': lstm_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'input_size': input_size,
                'output_size': output_size,
                'hidden_size': hidden_size,
                'loss':best_loss, #or val loss?
                'mse': mse,
                'mae': mae

            }, best_model_path)
            print(f"Model saved at epoch:{epoch}, mse:{mse}.", flush=True)

    return train_losses, val_losses


# The model should be called in your "training script".
# The following part is for unit test.
if __name__=="__main__":
    # Set up some example parameters
    input_size = 7
    hidden_size = 128
    output_size = 7
    batch_size = 4
    seq_len = 20
    lr = 0.001
    epoch = 1
    
    test_inputs = torch.randn(batch_size, seq_len, input_size)
    target_tensor = torch.randn(batch_size, seq_len, output_size)

    # Fixed input_len and target_len
    input_lens = torch.full((batch_size,), seq_len, dtype=torch.float32)  # Fixed seq_len
    target_lens = torch.full((batch_size,), seq_len, dtype=torch.float32)  # Fixed seq_len

    # Create TensorDataset
    dataset = TensorDataset(test_inputs, target_tensor, input_lens, target_lens)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #print(f"dataset shape: {test_inputs.shape,target_tensor.shape}")
    
    device = torch.device(f'cuda:1')
    best_model_path = "./models/test.pth"
    
    # Create model
    model = LSTMModel(output_seq = seq_len).to(device)

    # Create Optimizers
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Define Loss
    criterion = nn.MSELoss()

    Loss = running_train(dataloader,dataloader,model,opt,criterion,
                        n_epochs=epoch,best_model_path=best_model_path,device=device)
    print(Loss)