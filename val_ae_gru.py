"""
Author: Junas
- Val/Predict script for autoencoder model
"""

from utils.util import gpu_device_check
from datas.dataloader import create_dataloaders, create_pred_inputs
from models.model_autoencoder_gru import *

from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from glob import glob
import os

# Load the autoencoder model
def load_model(model_path):
    # model_path = f"./ckpts/AutoEncoder/AE_lstm_model_{input_seq_len}_{output_seq_len}_{special_test_name}.pth"
    checkpoint = torch.load(model_path)

    # Retrieve model parameters from checkpoint
    training_loss = checkpoint['loss']
    mse = checkpoint['mse']
    print(training_loss)
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    hidden_size = checkpoint['hidden_size']
    #layer_size = checkpoint["layer_size"]

    # Create instances of Encoder and Decoder using loaded parameters
    encoder = Encoder(input_size, hidden_size, 4).to(device)
    decoder = AttnDecoder(hidden_size, output_size, 4).to(device)

    # Load saved state_dict into the models
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    return encoder, decoder


# Define 
def eval_epoch(valloader, encoder, decoder, device):
    encoder.eval()
    decoder.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for val_input, val_target, input_len, target_len in valloader:
            # move dataloader to gpu
            val_input = val_input.to(device)
            val_target = val_target.to(device)

            # forward (predicting)
            encoder_outputs, encoder_hidden = encoder(val_input, input_len)
            decoder_outputs, _, _ = decoder(val_input, encoder_outputs,encoder_hidden)

            pred = decoder_outputs.cpu().detach().numpy()
            target = val_target.cpu().detach().numpy()
            preds.append(pred)
            targets.append(target)

        angle_actual = []
        angle_pred = []

        for target, pred in zip(targets, preds):
            angle_actual.append(target[:, :, -1])  # shape: [batch_size, seq_len]
            angle_pred.append(pred[:, :, -1])      # shape: [batch_size, seq_len]

        angle_actual = np.concatenate(angle_actual)
        angle_pred = np.concatenate(angle_pred)
    return  angle_actual, angle_pred

# Helper function for moving average smoothing
def smooth_sequence(sequence, window):
    smoothed_sequence = np.zeros_like(sequence)
    for i in range(len(sequence)):
        start_idx = max(0, i - window + 1)
        smoothed_sequence[i]=np.mean(sequence[start_idx:i+1], axis=0)
    return smoothed_sequence

# Helper function for EMA smoothing
def smooth_ema(sequence, alpha):
    ema_sequence = np.zeros_like(sequence)
    ema_sequence[0] = sequence[0]  # initializing
    for t in range(1, len(sequence)):
        ema_sequence[t] = alpha * sequence[t] + (1 - alpha) * ema_sequence[t - 1]
    return ema_sequence

def smooth_transition_linear(seq1, seq2, overlap=5):
    """
    use linear interpolation to smooth connecting part of seq1 & seq2
    Args:
        seq1: first seq,  shape=(length1, feature_dim)
        seq2: second seq, shape=(length2, feature_dim)
        overlap: length of connecting part
    Returns:
        new_seq1, new_seq2: seq after smoothing
    """
    transition = np.linspace(0, 1, overlap)[:, None]  # smoothing weight (overlap, 1)
    seq1[-overlap:] = (1 - transition) * seq1[-overlap:] + transition * seq2[:overlap]
    return seq1, seq2


# Define prediction epoch for curve (proposed strategy)
def pred_epoch(valloader, encoder, decoder, device, predict_len, steps, smooth_factor=0.8):
    encoder.eval()
    decoder.eval()
    
    val_input, val_target, input_len, target_len = valloader
    input_array = val_input.squeeze(0).cpu().numpy()  # save all results


    end = int(input_len)
    with torch.no_grad():  # 禁用梯度計算
        for _ in range(predict_len//steps+1):
            smoothed_input_array = smooth_ema(input_array, smooth_factor)
            #smoothed_input_array = smoothed_input_array[:end]
            current_input = torch.tensor(smoothed_input_array[-input_len:], dtype=torch.float32).unsqueeze(0).to(device)  # (1, 20, feature_dim)
            
            # 1. 模型進行預測
            # forward (predicting)
            encoder_outputs, encoder_hidden = encoder(current_input, input_len)
            decoder_outputs, _, _ = decoder(current_input, encoder_outputs,encoder_hidden)
            
            # 2. 提取前steps個預測點
            predicted_points = decoder_outputs[:,:steps,:].squeeze(0).cpu().numpy()  # (1, 5, feature_dim)
            # s1,s2 = smooth_transition_linear(smoothed_input_array, predicted_points,overlap=5)
            # s2 = s2[:steps, :]
            input_array = np.vstack([smoothed_input_array, predicted_points])
            end+=steps

            
    input_array= smooth_ema(input_array, smooth_factor)
    res_array = np.array(input_array)
    return res_array[:,-1]


if __name__=="__main__":
    model_path = "./ckpts/AutoEncoder/AE_gru_model_gru_aug_L4_4-30.pth"
    val_plot = False
    pred_plot = True
    pred_plot_strat2 = False
    # check GPU
    device = gpu_device_check(1)

    # generate encoder, decoder model
    encoder, decoder = load_model(model_path)
    
    ## 測試輸入不同長度的模型效果
    # seqs = [5,10,15,20,25,30]
    # for seq in seqs:
    #     # load data and sample with sliding windows
    #     train_dataloader, val_dataloader, input_size, output_size, scaler = create_dataloaders("raw_dataset",seq,seq,1,1)

    #     # defined evaluation function
    #     angle_actual, angle_pred = eval_epoch(val_dataloader, encoder, decoder, device)

    #     #angle_actual, angle_pred = actual[:,:,3], pred[:,:,3]

    #     mse = mean_squared_error(angle_actual.flatten(),angle_pred.flatten())
    #     mae = mean_absolute_error(angle_actual.flatten(),angle_pred.flatten())
    #     print(f"Mean Square Error: {mse}",f"Mean Absolute Error: {mae}")

    if val_plot:
        # 繪製 "angle" 的預測結果與實際值（每筆sequence）
        num_samples = angle_actual.shape[0]
        seq_length = angle_actual.shape[1]
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

        result_path = './demos/AutoEncoder/Result_gru_raw_L4_4-30.jpg'
        plt.savefig(result_path, format='jpg')
        print(f"Result curve saved to: {result_path}")

    if pred_plot:
        gt_list = glob("./datas/validation/**_augmented.csv")

        all_mses = []
        for in_len in range(10,31,10):
            #in_len = 8
            steps = in_len//2
            
            train_dataloader, val_dataloader, input_size, output_size, scaler = create_pred_inputs("aug_dataset",in_len,in_len,1,1)
            
            
            i=0
            mses = []
            for first_batch in val_dataloader:
                gt = pd.read_csv(gt_list[i],index_col=0)

                total_len = len(gt)
                predict_len = total_len-in_len
                
                res = pred_epoch(first_batch,encoder,decoder,device,predict_len,steps)
                #res = smooth_ema(res,0.8)
                res = res[:total_len]
                #res = smooth_sequence(res, 5)
                
                #gt = gt[["solr_value","temp_value","humd_value","angle"]].values
                gt = gt[["solr_value","temp_value","humd_value","cumulative_solr_value","cumulative_temp_value","angle"]].values
                gt = scaler.transform(gt)
                gt_ang = gt[:,-1]
                mse = mean_squared_error(gt_ang[in_len:],res[in_len:])
                mse = round(mse,3)
                mses.append(mse)
                print(f'Prediction MSE: {mse}')

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
                result_path = f'./curve_EMA/shortest/gif/{data_name}_{in_len}.jpg'
                plt.figure(figsize=(8,8))
                
                plt.plot(gt_ang,label="Ground Truth")
                plt.plot(res,label="Prediction")
                #plt.plot(gt_ang[:in_len],label="Input Sequence",color="r")
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



                #plt.title(f'Prediction Results')
                plt.xlabel('')
                plt.ylabel('')
                #plt.legend()
                plt.tight_layout()

                plt.savefig(result_path,format='jpg')
                print(f"Result curve saved to: {result_path}")

                # 要寫入的資料（作為字典）
                data = {
                    "GT": gt_ang,
                    "FC": res,
                }

                # 將資料轉換成 DataFrame
                df = pd.DataFrame(data)

                # 將 DataFrame 寫入 CSV

                df.to_csv(f'./curve_EMA/shortest/res/AE_{in_len}_{data_name}.csv', index=False, encoding="utf-8")
                i+=1
            print(f"Average MSE: {sum(mses)/len(mses)}")
        df_mse = pd.DataFrame(all_mses)
        df_mse.to_csv(f'./curve_EMA/shortest/AE_overall_mses.csv', index=False, encoding="utf-8")

    # 更新的策略，固定預測的step，但沒有使用在論文內
    if pred_plot_strat2:
        #gt = pd.read_csv('./datas/validation/set6_augmented.csv',index_col=0)
        #gt = pd.read_csv('./datas/training/set5_augmented.csv',index_col=0)
        gt_list = glob("./datas/validation/**_augmented.csv")

        all_mses = []
        for in_len in range(8,65,1):
            #in_len = 8
            steps = 8
            
            train_dataloader, val_dataloader, input_size, output_size, scaler = create_pred_inputs("aug_dataset",in_len,in_len,1,1)
            
            
            i=0
            mses = []
            for first_batch in val_dataloader:
                gt = pd.read_csv(gt_list[i],index_col=0)

                total_len = len(gt)
                predict_len = total_len-in_len
                
                res = pred_epoch(first_batch,encoder,decoder,device,predict_len,steps)
                #res = smooth_ema(res,0.8)
                res = res[:total_len]
                #res = smooth_sequence(res, 5)
                
                #gt = gt[["solr_value","temp_value","humd_value","angle"]].values
                gt = gt[["solr_value","temp_value","humd_value","cumulative_solr_value","cumulative_temp_value","angle"]].values
                gt = scaler.transform(gt)
                gt_ang = gt[:,-1]
                mse = mean_squared_error(gt_ang[in_len:],res[in_len:])
                mse = round(mse,3)
                mses.append(mse)
                print(f'Prediction MSE: {mse}')

                plt.rcParams['font.size'] = 24  # 字體大小
                plt.rcParams['lines.linestyle'] = '-'  # 全局線
                plt.rcParams['lines.marker'] = 'o'  # 全局標記
                #plt.rcParams['lines.linewidth'] = 2  # 設置所有粗度為 2 像素
                plt.rcParams['axes.linewidth'] = 1.5  # 設置邊框的粗度為 2 像素
                plt.rcParams['xtick.major.size'] = 10  # X 軸主刻度的長度
                plt.rcParams['xtick.major.width'] = 1.5  # X 軸主刻度的寬度
                plt.rcParams['ytick.major.size'] = 10  # Y 軸主刻度的長度
                plt.rcParams['ytick.major.width'] = 1.5  # Y 軸主刻度的寬度
                plt.rcParams['xtick.direction'] = 'in'  # X 軸刻度線方向
                plt.rcParams['ytick.direction'] = 'in'  # Y 軸刻度線方向

                data_name = os.path.basename(gt_list[i]).split('.')[0]
                result_path = f'./demos/AutoEncoder/gif_new_8/{data_name}_{in_len}.jpg'
                plt.figure(figsize=(8,8))
                
                plt.plot(gt_ang,label="Ground Truth")
                plt.plot(res,label="Prediction")
                plt.plot(gt_ang[:in_len],label="Input Sequence",color="r")
                ax = plt.gca()
                ax.spines['left'].set_position(('data', 0))  # 左邊框移到 x 軸 0 點
                ax.spines['right'].set_position(('data', len(res)))  # 右邊框移到 x 軸 300 點
                ax.set_xlim(0, len(res))  # 限制 x 軸範圍
                ax.set_ylim(0, 1.2)  # 限制 y 軸範圍


                #plt.title(f'Prediction Results')
                plt.xlabel('')
                plt.ylabel('')
                plt.legend()
                plt.tight_layout()

                plt.savefig(result_path,format='jpg')
                print(f"Result curve saved to: {result_path}")

                # 要寫入的資料（作為字典）
                data = {
                    "GT": gt_ang,
                    "FC": res,
                }

                # 將資料轉換成 DataFrame
                df = pd.DataFrame(data)
                # 將 DataFrame 寫入 CSV
                df.to_csv(f'./curve_EMA/gif_8/AE_{in_len}_{data_name}.csv', index=False, encoding="utf-8")
                
                i+=1
            all_mses.append(mses)
            print(f"Average MSE: {sum(mses)/len(mses)}")
        df_mse = pd.DataFrame(all_mses)
        df_mse.to_csv(f'./curve_EMA/overall/AE_overall_mses_{steps}.csv', index=False, encoding="utf-8")

