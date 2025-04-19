"""
Author: Junas
將曲線用sliding window方式抽樣並在每一個batch padding成一樣長度
"""

import glob
import pandas as pd
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


def load_data(path):
    files = glob.glob(path)
    df_list = []
    
    for file in files:
        #print(file)
        df = pd.read_csv(file,index_col=0)
        df_list.append(df)
    
    return df_list

def feature_select(DFS):
    raw_features = []
    cum_features = []
    aug_features = []

    #features selection
    for DF in DFS:
        features = DF[["solr_value","temp_value","humd_value","cumulative_solr_value","cumulative_humd_value","cumulative_temp_value","angle"]].values
        aug_features.append(features[:,[0,1,2,3,5,6]])
        raw_features.append(features[:,[0,1,2,-1]])
        cum_features.append(features[:,[3,4,5,-1]])

    return {"raw_dataset":raw_features,"cum_dataset":cum_features,"aug_dataset":aug_features}


#todo: leave out whole curve to actual test
def create_sample_set(data_list, min_window_size, max_window_size, sample_steps, scaler=None):
    # Fit and transform scaler on training data, transform validation data
    # fix sampling whole concatenated curve
    data_fit = np.concatenate(data_list,axis=0)
    transformed_data_list = []
    if scaler is None:
        scaler = MinMaxScaler().fit(data_fit)
        for data in data_list:
            data = scaler.transform(data)
            transformed_data_list.append(data)
    else:
        for data in data_list:
            data = scaler.transform(data)
            transformed_data_list.append(data)

    x, y = [], []
    for t_data in transformed_data_list:
        for j in range(min_window_size, max_window_size+2,2): #window size scaling as step=2
            window_size = j
            for i in range(0,len(t_data) - window_size - window_size + 1, sample_steps):
                x.append(t_data[i:i + window_size])
                y.append(t_data[i + window_size :i + window_size + window_size])
    return x,y,scaler #list
    
def create_pred_sample_set(data_list, min_window_size, max_window_size, sample_steps, scaler=None):
    # Fit and transform scaler on training data, transform validation data
    # fix sampling whole concatenated curve
    data_fit = np.concatenate(data_list,axis=0)
    transformed_data_list = []
    if scaler is None:
        scaler = MinMaxScaler().fit(data_fit)
        for data in data_list:
            data = scaler.transform(data)
            transformed_data_list.append(data)
    else:
        for data in data_list:
            data = scaler.transform(data)
            transformed_data_list.append(data)

    x, y = [], []
    for t_data in transformed_data_list:
        for j in range(min_window_size, max_window_size+2,2): #window size scaling as step=2
            window_size = j
            x.append(t_data[:window_size])
            y.append(t_data[window_size : window_size + window_size])
    return x,y,scaler #list

class SequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

def collate_fn(batch):
    # 取得 x 和 y
    x_batch = [item[0].clone().detach() for item in batch]
    y_batch = [item[1].clone().detach() for item in batch]

    # 取得每个序列的长度
    x_lengths = torch.tensor([len(x) for x in x_batch])
    y_lengths = torch.tensor([len(y) for y in y_batch])

    # 將 x 和 y 使用 pad_sequence 填充到相同的長度
    x_padded = pad_sequence(x_batch, batch_first=True, padding_value=-1)  # [batch_size, max_seq_len, num_features]
    y_padded = pad_sequence(y_batch, batch_first=True, padding_value=-1)  # [batch_size, max_output_len, num_features]

    return x_padded, y_padded, x_lengths, y_lengths

# def process_features(features_list, win_size, pred_size, steps, window_size_adj):
#     # Split each dataset into train and validation sets, then concatenate
#     data_list = 
#     train = np.concatenate(train_list, axis=0)
#     val = np.concatenate(val_list, axis=0)

#     # Fit and transform scaler on training data, transform validation data
#     scaler = MinMaxScaler()
#     train_normalized = scaler.fit_transform(train)
#     val_normalized = scaler.transform(val)

#     # Create datasets for training and validation
#     X_train, Y_train = create_dataset(train_normalized, win_size, pred_size, steps, window_size_adj)
#     X_val, Y_val = create_dataset(val_normalized, win_size, pred_size, steps, window_size_adj)

#     # Return dictionary with train/val datasets and scaler
#     return {
#         "train": [X_train, Y_train],
#         "val": [X_val, Y_val],
#         "scaler": scaler
#     }
#dataset preprocess
#todo: is scaling needed or reasonable


# # Split dataset into training and validation sets
# def split_train_val(X, Y, val_size=0.2):
#     # 使用 sklearn 的 train_test_split 進行資料切分
#     #X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_size, shuffle=True, random_state=42)
#     X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_size, shuffle=False)
#     return X_train, X_val, Y_train, Y_val





# main function
def create_dataloaders(feature_type:str, min_seq_len=10, max_seq_len=30, steps=1, batch_size=1):
    train_data = load_data("./datas/training/**augmented.csv")
    val_data = load_data("./datas/validation/**augmented.csv")
    
    #input_seq_len, output_seq_len, steps = 20,10, 1
    feature_key = feature_type  #"raw_dataset"/"cum_dataset"/"all_dataset"
    train_dataset = feature_select(train_data)[feature_key]
    val_dataset = feature_select(val_data)[feature_key]
    
    X_train, Y_train, scaler = create_sample_set(train_dataset, min_seq_len, max_seq_len, steps)
    X_val, Y_val, _ = create_sample_set(val_dataset, min_seq_len, max_seq_len, steps, scaler=scaler)
    
    # #feature size
    input_size, output_size = X_train[0].shape[1], X_train[0].shape[1]

    train_seq_set = SequenceDataset(X_train, Y_train)
    val_seq_set = SequenceDataset(X_val, Y_val)

    train_dataloader = DataLoader(train_seq_set, batch_size=batch_size, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_seq_set, batch_size=batch_size, collate_fn=collate_fn)


    return train_dataloader, val_dataloader, input_size, output_size, scaler

def create_pred_inputs(feature_type:str, min_seq_len=10, max_seq_len=30, steps=1, batch_size=1):
    train_data = load_data("./datas/training/**augmented.csv")
    val_data = load_data("./datas/validation/**augmented.csv")
    
    #input_seq_len, output_seq_len, steps = 20,10, 1
    feature_key = feature_type  #"raw_dataset"/"cum_dataset"/"all_dataset"
    train_dataset = feature_select(train_data)[feature_key]
    val_dataset = feature_select(val_data)[feature_key]

    X_train, Y_train, scaler = create_pred_sample_set(train_dataset, min_seq_len, max_seq_len, steps)
    X_val, Y_val, _ = create_pred_sample_set(val_dataset, min_seq_len, max_seq_len, steps, scaler=scaler)

    # #feature size
    input_size, output_size = X_train[0].shape[1], X_train[0].shape[1]

    train_seq_set = SequenceDataset(X_train, Y_train)
    val_seq_set = SequenceDataset(X_val, Y_val)

    train_dataloader = DataLoader(train_seq_set, batch_size=batch_size, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_seq_set, batch_size=batch_size, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, input_size, output_size, scaler

if __name__=="__main__":
    ts,vs,inps,outps,scaler = create_dataloaders("aug_dataset",10,10,1,256) #default settings
    #print(ts.shape)
    train_count = 0
    val_count = 0
    for x,y,xl,yl in ts:
        train_count+=x.shape[0]
        print(x.shape)
    
    for x,y,xl,yl in vs:
        val_count+=x.shape[0]
        print(x.shape)

    print(f'Train: {train_count}, Val: {val_count}')
    