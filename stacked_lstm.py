"""
Author: Junas
-Train/Val/Predict script for lstm model
"""

#import torch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

#import helper functions (local)
from models.model_lstm import *
from datas.dataloader import create_pred_inputs, create_dataloaders
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
import os

# Load trained model
def load_model(model_path):
    # model_path = f"./ckpts/AutoEncoder/AE_lstm_model_{input_seq_len}_{output_seq_len}_{special_test_name}.pth"
    checkpoint = torch.load(model_path)

    # Retrieve model parameters from checkpoint
    training_loss = checkpoint['loss']
    print(training_loss)
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    hidden_size = checkpoint['hidden_size']
    #layer_size = checkpoint["layer_size"]

    # Create instances of Encoder and Decoder using loaded parameters
    model = LSTMModel(input_size = input_size, hidden_size=hidden_size, output_size=output_size, output_seq = 20).to(device)

    # Load saved state_dict into the models
    model.load_state_dict(checkpoint['model'])

    return model

# Helper function for EMA process
def smooth_ema(sequence, alpha):
    ema_sequence = np.zeros_like(sequence)
    ema_sequence[0] = sequence[0]  # Initializing
    for t in range(1, len(sequence)):
        ema_sequence[t] = alpha * sequence[t] + (1 - alpha) * ema_sequence[t - 1]
    return ema_sequence


# Predicting the curve with proposed "Strategy"
def pred_epoch(valloader, model, device, predict_len, steps, smooth_factor=0.8):
    model.eval()

    val_input, val_target, input_len, target_len = valloader
    input_array = val_input.squeeze(0).cpu().numpy()  # 用來存儲所有的預測結果

    end = int(input_len)
    with torch.no_grad():
        for _ in range(predict_len//steps+1):
            # move dataloader to gpu
            smoothed_input_array = smooth_ema(input_array, smooth_factor)
            current_input = torch.tensor(smoothed_input_array[-input_len:], dtype=torch.float32).unsqueeze(0).to(device)  # (1, 20, feature_dim)
            
            #initialize
            h_0 = torch.zeros(model.num_layers, 1, model.hidden_size, device=device)
            c_0 = torch.zeros(model.num_layers, 1, model.hidden_size, device=device)
            state = (h_0,c_0)

            # forward (predicting)
            #print(current_input.shape, state[0].shape)
            lstm_outputs, state = model(current_input,state)
            #print(current_input.shape, state[0].shape)
            

            predicted_points = lstm_outputs[:,:steps,:].squeeze(0).cpu().numpy()
            input_array = np.vstack([smoothed_input_array, predicted_points])
            end+=steps

    input_array= smooth_ema(input_array, smooth_factor)
    res_array = np.array(input_array)
    return res_array[:,-1]


# Main part of controlling train/val/test/pred process
if __name__ == "__main__":
    # cli argparse (嫌麻煩請依照每個task註解掉不需要的)
    parser = argparse.ArgumentParser(description="Run model with specified configuration file.")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the configuration YAML file.")
    parser.add_argument("--gpu", type=int, required=True, help="Index of GPU.")
    parser.add_argument("--dataset_type", type=str, required=True, help="raw_dataset/cum_dataset/aug_dataset")
    
    #Process control
    lstm_training = False
    lstm_validation = False
    lstm_predicting = True

    if lstm_training:        
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
            best_model_path = f"./ckpts/StackedLSTM/Stacked_LSTM_{config_name}_{min_seq_len}-{max_seq_len}.pth"  # Model save dir
            #minor settings:
            steps=1

            #generate Datasets
            train_dataloader, val_dataloader, input_size, output_size, scaler = create_dataloaders(args.dataset_type,min_seq_len,max_seq_len,steps,batch_size)

            # Create Encoder, Decoder
            model = LSTMModel(input_size, hidden_size, output_size, num_layers, dropout=0.1,output_seq=max_seq_len, device=device).to(device)

            # Create Optimizers
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Define Loss
            criterion = nn.MSELoss()

            # Training
            print(f"Training model: {best_model_path}", flush=True)
            TrainLoss, ValidLoss = running_train(train_dataloader,val_dataloader,model,opt,criterion,
                            n_epochs=epoch,best_model_path=best_model_path,device=device)

            #plot loss curve and save to dir
            plt.figure()
            plt.plot(TrainLoss, label="Training Loss")
            plt.plot(ValidLoss, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss Curve")
            plt.legend()
            loss_curve_path = f"./demos/StackedLSTM/Stacked_LSTM_{config_name}_{min_seq_len}_{max_seq_len}.png"
            plt.savefig(loss_curve_path)

            print(f"Loss curve saved to: {loss_curve_path}")


        torch.cuda.empty_cache()

    if lstm_validation:
        args = parser.parse_args()
        # check GPU
        device = gpu_device_check(args.gpu)

        # Load the model
        model_path = f"./ckpts/StackedLSTM/Stacked_LSTM_aug_l_30-30.pth"  # Saved model path
        lstm_model = load_model(model_path)

        # 預測結果
        actual, pred = evaluate_model(lstm_model, val_dataloader)
        angle_actual, angle_pred = actual[:,:,3], pred[:,:,3]
        # print(angle_pred)
        
        
        num_samples = angle_actual.shape[0]
        seq_length = angle_actual.shape[1]

        # Calculate MSE
        mse = mean_squared_error(actual.flatten(),pred.flatten())
        mae = mean_absolute_error(actual.flatten(),pred.flatten())
        print(f"Mean Square Error: {mse}",f"Mean Absolute Error: {mae}")
        
        # 繪製 "angle" 的預測結果與實際值 (每個sequence)
        if lstm_val_plot:

            # 設定子圖的行列數量（例如，4行4列）
            num_rows = int(np.ceil(np.sqrt(num_samples)))
            num_cols = int(np.ceil(num_samples / num_rows))

            plt.figure(figsize=(num_cols*5, num_rows * 4))
            for i in tqdm(range(num_samples)):
                plt.subplot(num_rows, num_cols, i + 1)
                plt.plot(angle_actual[i], label='Actual Angle')
                plt.plot(angle_pred[i], label='Predicted Angle')
                plt.title(f'Sample {i + 1}: Actual vs Predicted Angle')
                plt.xlabel('Time Step')
                plt.ylabel('Angle')
                #plt.legend()
                plt.tight_layout()

            plt.savefig(f'result_gridplot_{input_seq_len}_{output_seq_len}_{special_test_name}.jpg', format='jpg')

        
    # Predict the curve with proposed "Strategy"
    if lstm_predicting:
        model_path = f"./ckpts/StackedLSTM/Stacked_LSTM_aug_l_30-30.pth"  # Saved model name
        device = gpu_device_check(0)
        
        #read gt data
        gt_list = glob.glob("./datas/validation/**_augmented.csv")

        # Load the trained LSTM model
        lstm_model = load_model(model_path).to(device)
        
        # set the input sequence
        min_len = 30
        max_len = 30
        steps = min_len//2
        
        train_dataloader, val_dataloader, input_size, output_size, scaler = create_pred_inputs("aug_dataset",min_len,max_len,1,1)
        
        i=0
        mses = []
        for first_batch in val_dataloader:
            gt = pd.read_csv(gt_list[i],index_col=0)
            
            total_len = len(gt)
            predict_len = total_len-min_len
            res = pred_epoch(first_batch,lstm_model,device,predict_len,steps)
            #res = smooth_ema(res,0.8)
            res = res[:total_len]
            #res = smooth_sequence(res, 5)

            #gt = gt[["solr_value","temp_value","humd_value","angle"]].values
            gt = gt[["solr_value","temp_value","humd_value","cumulative_solr_value","cumulative_temp_value","angle"]].values
            gt = scaler.transform(gt)
            gt_ang = gt[:,-1]
            mse = mean_squared_error(gt_ang[min_len:],res[min_len:])
            mse = round(mse,3)
            mses.append(mse)
            print(f'Prediction MSE: {mse}')

            #繪圖參數
            plt.rcParams['font.size'] = 24  # 字體大小
            plt.rcParams['lines.linestyle'] = '-'  # 全局線
            plt.rcParams['lines.marker'] = 'o'  # 全局標記
            #plt.rcParams['lines.linewidth'] = 2  # 設置所有粗度為 2 像素
            plt.rcParams['axes.linewidth'] = 1.5  # 設置邊框的粗度為 2 像素
            plt.rcParams['xtick.major.size'] = 10  # X 軸主刻度的長度
            plt.rcParams['xtick.major.width'] = 1.5  # X 軸主刻度的寬度
            plt.rcParams['xtick.minor.size'] = 6 # X 軸次要刻度（Minor tick）的長度
            plt.rcParams['xtick.minor.width'] = 1  # X 軸次要刻度（Minor tick）的寬度
            plt.rcParams['ytick.major.size'] = 10  # Y 軸主刻度的長度
            plt.rcParams['ytick.major.width'] = 1.5  # Y 軸主刻度的寬度
            plt.rcParams['xtick.direction'] = 'in'  # X 軸刻度線方向
            plt.rcParams['ytick.direction'] = 'in'  # Y 軸刻度線方向

            data_name = os.path.basename(gt_list[i]).split('.')[0]
            result_path = f'./demos/StackedLSTM/aug_l_{max_len}_{data_name}.jpg'
            plt.figure(figsize=(8,8))
            
            plt.plot(gt_ang,label="Ground truth")
            plt.plot(res,label="Predicted")
            #plt.title(f'Prediction Results')
            plt.xlabel('')
            plt.ylabel('')
            
            ax = plt.gca()
            ax.spines['left'].set_position(('data', 0))  # 左邊框移到 x 軸 0 點
            ax.spines['right'].set_position(('data', len(res)))  # 右邊框移到 x 軸長度
            ax.set_xlim(0, len(res))  # 限制 x 軸範圍
            ax.set_ylim(0, 1.2)  # 限制 y 軸範圍

            major_interval = 24  # 每 12 小時一個 major tick (30 分鐘一個 time step, 12小時=24 time steps)
            minor_interval = 2


            # 設定 X 軸的 major ticks
            ax.set_xticks(np.arange(0, len(gt_ang), major_interval))
            # 設定 Minor ticks（每 1 小時）
            ax.set_xticks(np.arange(0, len(gt_ang), minor_interval), minor=True)
            
            # 設定 X 軸的標籤顯示，這裡轉換成小時數
            ax.set_xticklabels([f"{int(i/2)}" for i in np.arange(0, len(gt_ang), major_interval)])

            
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '' if x == 0 else f'{x:.1f}'))
            
            #plt.legend()
            plt.tight_layout()
            plt.savefig(result_path,format='jpg')
            print(f"Result curve saved to: {result_path}")

            data = {
                "GT": gt_ang,
                "FC": res,
            }

            # 將資料轉換成 DataFrame
            df = pd.DataFrame(data)

            # 將 DataFrame 寫入 CSV
            df.to_csv(f'./curve_EMA/LSTM_{max_len}_{data_name}.csv', index=False, encoding="utf-8")


            i+=1
        print(f"Average MSE: {sum(mses)/len(mses)}")