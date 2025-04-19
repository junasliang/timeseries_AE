"""
Author: Junas
- Training script for autoencoder model
"""

#import torch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

#import helper functions (local)
from models.model_autoencoder_gru import *
from datas.dataloader import create_dataloaders
from utils.util import gpu_device_check
from utils.cfg_parser import load_cfg_list, parse_cfg

#import sklearn for dataset preparing, mse calculaiton
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

#import tool modules
import argparse
import numpy as np
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime



def evaluate_model():
    raise NotImplementedError
    
def predict_model():
    raise NotImplementedError


if __name__=="__main__":
    # cli argparse
    parser = argparse.ArgumentParser(description="Run model with specified configuration file.")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the configuration YAML file.")
    parser.add_argument("--gpu", type=int, required=True, help="Index of GPU.")
    parser.add_argument("--dataset_type", type=str, required=True, help="raw_dataset/cum_dataset/all_dataset")
    
    args = parser.parse_args()

    # check GPU
    device = gpu_device_check(args.gpu)

    #read config list:
    config_list = load_cfg_list(args.cfg) #input path for other yaml, default='../cfg.yaml'

    #Start training
    for cfg in config_list:
        #log start time
        print(datetime.now())
        #parse training settings:
        config_name, min_seq_len, max_seq_len, val_ratio, batch_size, hidden_size, epoch, lr, weight_decay, teacher_forcing, num_layers = parse_cfg(cfg)
        best_model_path = f"./ckpts/AutoEncoder/AE_gru_model_{config_name}_{min_seq_len}-{max_seq_len}.pth"  # Model save dir
        
        #minor settings:
        steps=1

        #generate Datasets
        train_dataloader, val_dataloader, input_size, output_size, scaler = create_dataloaders(args.dataset_type,min_seq_len,max_seq_len,steps,batch_size)

        # Create Encoder, Decoder
        encoder = Encoder(input_size, hidden_size,num_layers).to(device)
        decoder = AttnDecoder(hidden_size, output_size,num_layers).to(device)

        # Create Optimizers
        encoder_opt = optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
        decoder_opt = optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)

        # Create LR schedulers
        encoder_scheduler = CosineAnnealingLR(encoder_opt, T_max=epoch)
        decoder_scheduler = CosineAnnealingLR(decoder_opt, T_max=epoch)

        # Define Loss
        criterion = nn.MSELoss(reduction="none")

        # Training
        print(f"Training model: {best_model_path}", flush=True)
        TrainLoss, ValidLoss  = running_train(train_dataloader,val_dataloader,encoder,decoder,encoder_opt,decoder_opt,
                                                encoder_scheduler,decoder_scheduler,criterion,teacher_forcing=teacher_forcing,
                                                n_epochs=epoch,best_model_path=best_model_path,device=device)
        
        # Save losses to CSV
        loss_csv_path = f"./demos/AutoEncoder/AE_gru_loss_{config_name}_{min_seq_len}_{max_seq_len}.csv"
        with open(loss_csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Epoch", "Training Loss", "Validation Loss"])
            for i, (train_loss, valid_loss) in enumerate(zip(TrainLoss, ValidLoss), start=1):
                writer.writerow([i, train_loss, valid_loss])
        print(f"Losses saved to CSV: {loss_csv_path}")

        #plot loss curve and save to dir
        plt.figure()
        plt.plot(TrainLoss, label="Training Loss")
        plt.plot(ValidLoss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        loss_curve_path = f"./demos/AutoEncoder/AE_gru_loss_curve_{config_name}_{min_seq_len}_{max_seq_len}.png"
        plt.savefig(loss_curve_path)

        print(f"Loss curve saved to: {loss_curve_path}")

    torch.cuda.empty_cache()
