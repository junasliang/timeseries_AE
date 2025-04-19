import numpy as np 
from numpy import log
import pandas as pd
import os

import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA


from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf

from pmdarima import auto_arima

import matplotlib.pyplot as plt
from pylab import rcParams

#import local helper funciton
from datas.dataloader import load_data, feature_select

import random


#data prep
def load_dataframe():
    train_data = load_data("./datas/training/**augmented.csv")
    test_data = load_data("./datas/validation/**augmented.csv")
    train_data = [train_data[0],train_data[2],train_data[4],train_data[5],train_data[6],train_data[7],train_data[1],train_data[3]]
    test_data = [test_data[0],test_data[2],test_data[1]]
    #print(len(train_data),len(val_data))
    
    #leave one out
    # all_data = train_data+val_data
    #random.shuffle(train_data)
    # test_data = all_data.pop()
    
    all_dataframe = pd.concat(train_data+test_data, ignore_index=True)
    all_dataframe = all_dataframe[["solr_value","temp_value","humd_value","cumulative_solr_value","cumulative_temp_value","angle"]]
    train_dataframe = pd.concat(train_data, ignore_index=True)
    train_dataframe = train_dataframe[["solr_value","temp_value","humd_value","cumulative_solr_value","cumulative_temp_value","angle"]]
    test_dataframe = pd.concat(test_data, ignore_index=True)
    test_dataframe = test_dataframe[["solr_value","temp_value","humd_value","cumulative_solr_value","cumulative_temp_value","angle"]]
    
    #scaler = MinMaxScaler()
    #train_dataframe = pd.DataFrame(scaler.fit_transform(train_dataframe), columns=train_dataframe.columns, index=train_dataframe.index)
    
    # val_datas=[]
    # for data in val_data:
    #     data = data[["solr_value","temp_value","humd_value","cumulative_solr_value","cumulative_humd_value","cumulative_temp_value","angle"]]
    #     test_dataframe = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)
    #     val_datas.append(test_dataframe)
    # print(type(train_dataframe))
    #feature selection
    # train_dataset = feature_select(train_data)[feature_type]
    # val_dataset = feature_select(val_data)[feature_type]
    print(f'Dataframe shape: {train_dataframe.shape}')
    print(f'Dataframe feature: {train_dataframe.columns.tolist()}')
    return train_dataframe, test_dataframe, all_dataframe#, scaler

#data analysis adf
def adf_test(train_df):
    features = train_df.columns.tolist()
    for feature in features:
        adf = sm.tsa.stattools.adfuller(train_df[feature])
        print(f'ADF Statistic for {feature}: {adf[0]:f}')
        print(f'p-value: {adf[1]:f}')
        print(f'Critical Values:')
        for key, value in adf[4].items():
            print(f'\t{key}: {value:f}')

#data analysis OLS
def ols_test(train_df):
    train_df['const']=1 #Adding a constant
    train_df['diff_temp']=train_df['cumulative_temp_value'].diff() #First order differencing for acc temp
    #train_df['diff_humd']=train_df['cumulative_humd_value'].diff() #First order differencing for acc humid
    train_df['diff_solr']=train_df['cumulative_solr_value'].diff() #First order differencing for acc humid
    train_df['diff_angle']=train_df['angle'].diff() #First order differencing for value

    model1=sm.OLS(endog=train_df['angle'],exog=train_df[["solr_value","temp_value","humd_value","cumulative_solr_value","cumulative_humd_value","cumulative_temp_value","const"]])
    results1=model1.fit()
    print(results1.summary())

    #Fitting the model on the differenced data
    # model2=sm.OLS(endog=train_df['diff_angle'].dropna(),exog=train_df[["solr_value","temp_value","humd_value",'diff_temp','diff_humd','diff_solr','const']].dropna())
    # results2=model2.fit()
    # print(results2.summary())

#data analysis decomposition
def decomp(train_df):
    print(len(train_df))
    angs = train_df["angle"]
    decomposition = sm.tsa.seasonal_decompose(angs, model='additive', extrapolate_trend='freq',period=8)
    
    fig = decomposition.plot().set_size_inches(18,8)
    plt.tight_layout()
    plt.savefig("./sarimax_result/decomposed_result.png",dpi=300)
    print(f"image saved: ./sarimax_result/decomposed_result.png ")

    # #maybe useless: adf of diff angle
    # diff_ang = angs-angs.shift(1)
    # diff_ang = diff_ang.dropna()
    # # 創建新圖像對象並繪製差分圖
    # plt.figure(figsize=(16, 4))  # 新圖像對象
    # fig2 = diff_ang.plot()
    # fig2.set_xlim([0, len(diff_ang) - 1])
    # fig2.set_xticks(range(0, len(diff_ang), 100))
    # plt.title("First Difference of Angle")
    # plt.xlabel("Time")
    # plt.ylabel("Difference")
    # plt.tight_layout()
    # plt.savefig("./diff_result.png", dpi=300)
    # print(f"Image saved: ./diff_result.png")

#fft analysis: find seasonal period by fft method
def fft_seasonal(train_df):
    ang = train_df["angle"]
    fft = np.fft.fft(ang.dropna())
    freq = np.fft.fftfreq(len(fft))
    period = 1 / freq[np.argmax(np.abs(fft[1:])) + 1]
    print(f"FFT detected period: {period}")


#acf/pacf analysis: find seasonal period by observing acf/pacf
def period_selection(train_df):
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
    
    angs = train_df["angle"]
    
    # 計算 ACF 值
    acf_values = acf(angs, nlags=300)
    # 找到峰值的滯後點
    peaks, _ = find_peaks(acf_values, distance=5)  # 設定最小間隔避免多餘峰值
    #print("Peaks at lags:", peaks)

    # 繪製 ACF 圖並標註峰值
    fig_acf, ax_acf = plt.subplots(figsize=(24, 8))
    plot_acf(angs, lags=300, ax=ax_acf)
    ax_acf.plot(peaks, acf_values[peaks], 'o',color='red', label='Detected Peaks')  # 用紅色圓點標註峰值
    ax_acf.legend(loc='upper right')
    ax_acf.set_title("")
    ax_acf.set_xlabel("Lag (every 30 minutes)")  # x 軸標籤
    ax_acf.set_ylabel("Autocorrelation")   # y 軸標籤

    # 調整 y 軸貼齊 x 軸的 0 點，並設置右邊框與 x 軸 300 對齊
    ax_acf.spines['left'].set_position(('data', 0))  # 左邊框移到 x 軸 0 點
    ax_acf.spines['right'].set_position(('data', 300))  # 右邊框移到 x 軸 300 點
    #ax_acf.spines['top'].set_visible(False)  # 隐藏上边框
    ax_acf.set_xlim(0, 300)  # 限制 x 軸範圍
    
    plt.savefig("./sarimax_result/acf_with_peaks.png")
    

    # 繪製 PACF 圖
    fig_pacf, ax_pacf = plt.subplots(figsize=(16, 8))
    plot_pacf(angs, lags=300, method='ywm', ax=ax_pacf)
    ax_pacf.set_title("")
    plt.savefig("./sarimax_result/pacf.png")    
    
    periods = np.diff(peaks)
    seasonal_period = periods.mean()
    print("Estimated seasonal period (s):", seasonal_period)

    # 計算 PACF 值來估算 p
    pacf_values = pacf(angs, nlags=300, method='ywm')  # 設定計算的最大滯後
    #print("PACF values:", pacf_values)
    threshold_p = 1.96 / np.sqrt(len(angs))  # 95% 置信區間的閾值
    significant_p_lags = [i for i, val in enumerate(pacf_values) if abs(val) > threshold_p]
    p = significant_p_lags[1] if len(significant_p_lags) > 1 else 0  # 跳過第 0 點
    print("Estimated p:", p)

    # 計算 ACF 值來估算 q
    threshold_q = 1.96 / np.sqrt(len(angs))  # 95% 置信區間的閾值
    significant_q_lags = [i for i, val in enumerate(acf_values) if abs(val) > threshold_q]
    q = significant_q_lags[1] if len(significant_q_lags) > 1 else 0  # 跳過第 0 點
    print("Estimated q:", q)

    return {
        "p": p,
        "q": q,
        "seasonal_period": seasonal_period
    }

# auto-find arima order: find the order by minimizing AIC
# you should not input your validation data during this operation
def arimax_para(train_df, period=None): #might need input period
    ang = train_df["angle"]
    exog = train_df[["solr_value","temp_value","humd_value","cumulative_solr_value","cumulative_temp_value"]]
    stepwise_model = auto_arima(
        ang,
        exog,                         # do not use "exogenous="! fucking module error... 
        start_p=0, start_q=0,         # AR 和 MA 初始值
        max_p=10, max_q=10,           # AR 和 MA 最大階數
        m=period,                     # 季節性週期
        start_P=0, max_P=3,           # 季節 AR 初始值和最大階數
        start_Q=0, max_Q=3,           # 季節 MA 初始值和最大階數
        d=None, D=1,                  # 差分階數自動選擇，季節差分階數設為 1
        seasonal=True,                # 啟用季節性
        trace=True,                   # 打印每次嘗試的參數組合及其 AIC
        error_action='ignore',        # 忽略不可用參數組合
        suppress_warnings=True,       # 抑制警告信息
        stepwise=True                 # 使用逐步搜索來加速選擇
    )

    # print model summary
    print(stepwise_model.summary())
    p, d, q = stepwise_model.order
    P, D, Q, m = stepwise_model.seasonal_order
    
    return (p,d,q), (P,D,Q,m)
    
# "DEFINE" the sarimax model with best searched "ORDER"
# "FIT" the model with input datas to find "PARAMETERS/VARIABLES"
def build_arimax(train_df,val_df, n_ord,s_ord=None,input_len=0):
    target_len = len(val_df)-input_len
    alldf= pd.concat([train_df,val_df], ignore_index=True)
    train = alldf[:-target_len]
    val = alldf[-target_len:]
    ang = train["angle"]
    exog = train[["solr_value","temp_value","humd_value","cumulative_solr_value","cumulative_temp_value"]]
    arimax = sm.tsa.statespace.SARIMAX(
        ang,
        order=n_ord,
        seasonal_order=s_ord,
        exog=exog,
        enforce_stationarity=False,
        enforce_invertibility=False
        ).fit()
    print(arimax.summary())
    return arimax, train, val

"==========================================================================================================="
"""
# input data architecture:
# angle[0:20]
# env_data[0:len(angle_curve)]

# output data architecture:
# predicted angle[21:len(angle_curve)]
"""
# predict with input data, concate the initializing sequence to training data (not sure if needed)
def pred_arimax(model, train, target, input_len=0, idx=0):
    
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

    target_angle = target["angle"]
    target_exog = target[["solr_value","temp_value","humd_value","cumulative_solr_value","cumulative_temp_value"]]
    target["forecast"] = model.predict(len(train),len(train)+len(target)-1,exog=target_exog,dynamic=False)[0:]
    #train_df[["angle", "forecast"]][-70:].plot(figsize=(12,8),legend=False)
    
    mse = mean_squared_error(target_angle, target["forecast"])
    print(f"Mean Squared Error (MSE) between 'angle' and 'forecast': {mse}")
    
    plt.figure(figsize=(8,8))
    pred = pd.concat([train["angle"][-input_len:],target["forecast"]])
    pred = pred.reset_index(drop=True)
    gt = pd.concat([train["angle"][-input_len:],target["angle"]])
    gt = gt.reset_index(drop=True)
    
    plt.plot(gt,label="Ground truth")
    plt.plot(pred,label="Predicted")

    data = {
                "GT": gt,
                "FC": pred,
            }

    # 將資料轉換成 DataFrame
    df = pd.DataFrame(data)

    # 將 DataFrame 寫入 CSV
    #df.to_csv(f'./curve_EMA/SARIMAX_{input_len}_{idx}.csv', index=False, encoding="utf-8")

    #plt.title("SARIMAX(AIC) Forecasting")
    plt.xlabel("")
    plt.ylabel("")
    #plt.legend()
    
    # 調整 y 軸貼齊 x 軸的 0 點，並設置右邊框與 x 軸 300 對齊
    ax = plt.gca()
    ax.spines['left'].set_position(('data', 0))  # 左邊框移到 x 軸 0 點
    ax.spines['right'].set_position(('data', len(pred)))  # 右邊框移到 x 軸 300 點
    ax.set_xlim(0, len(pred))  # 限制 x 軸範圍
    ax.set_ylim(0, 1.2)  # 限制 y 軸範圍
    
    plt.tight_layout()
    

    #plt.savefig(f"./sarimax_result/set{idx}_forecast{input_len}_aic.png")

# helper function: plot the angle curves (concated/single)
def plot_cat(train_df, one_set=False):
    plt.rcParams['font.size'] = 30  # 字體大小
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

    if one_set:
        train_df["angle"][:117].plot(figsize=(8,8),linewidth=2)
        name = "original_data_set1"
    else:
        plt.rcParams['lines.marker'] = ""  # 全局標記
        train_df["angle"].plot(figsize=(24,8),linewidth=2)
        name = "original_data"

    #plt.title("Sets of Flowering Angle")
    plt.xlabel("Flowering period (Hour)")
    plt.ylabel("Flowering angle (Degree)")

    ax = plt.gca()
    ax.spines['left'].set_position(('data', 0))  # 左邊框移到 x 軸 0 點
    ax.spines['right'].set_position(('data', len(train_df)))  # 右邊框移到 x 軸 300 點
    ax.set_xlim(0, len(train_df))  # 限制 x 軸範圍
    ax.set_ylim(0,)  # 限制 y 軸範圍
    
    major_interval = 48  # 每 12 小時一個 major tick (30 分鐘一個 time step, 12小時=24 time steps)
    minor_interval = 4

    # 設定 X 軸的 major ticks
    ax.set_xticks(np.arange(0, len(train_df), major_interval))
    # 設定 Minor ticks（每 1 小時）
    ax.set_xticks(np.arange(0, len(train_df), minor_interval), minor=True)

    ax.set_yticks(np.arange(0, 150, 20))  # 設定 major tick（每 10 單位）
    
    # 設定 X 軸的標籤顯示，這裡轉換成小時數
    ax.set_xticklabels([f"{int(i/2)}" for i in np.arange(0, len(train_df), major_interval)])

    # ax.yaxis.set_major_locator(MultipleLocator(y_major_interval))
    # ax.yaxis.set_minor_locator(MultipleLocator(y_minor_interval))
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '' if x == 0 else f'{x:.0f}'))


    plt.tight_layout()
    plt.savefig(f"./sarimax_result/{name}.png")
    print(f"image save at: ./sarimax_result/{name}.png")


# main function, please comment out needless block
# 主程式碼，依照需求註解掉不需要用的部分
if __name__ == "__main__":

#### Basic analysis functions #################
    # train, test, all_d= load_dataframe()
    # p = period_selection(train)["seasonal_period"]
    # adf_test(train)
    # ols_test(train)
    # fft_seasonal(train)
    # decomp(train)


#### Find average seasonal period (請修改load_dataframe內的random order) #################
    # p=0
    # for i in range(5):
    #     train, test, scl = load_dataframe()
    #     p += period_selection(train)["seasonal_period"]
    # print(f"average period (hour): {p/10}")


#### AIC search for best ORDER #################
    # train, test, all_d= load_dataframe()
    # for i in range(5):
    #     train, test, sc = load_dataframe() #"raw_dataset"/"cum_dataset"/"all_dataset"
    #     order, s_order = arimax_para(train,24) # search for arimax parameters
    #     print(order,s_order)
    #     with open("arimax_results_48.txt", "a") as f:
    #         f.write(f"Iteration {i}: Order: {order}, Seasonal Order: {s_order}\n")


#### Build and fit the model #################
    # model, t, v = build_arimax(train,order,s_order)


#### Predict with the model #################
    # ini_len = 30
    # for i in range(3):
    #     test = tests[i]
    #     model, t, v = build_arimax(train,test,(3,0,2),(0,1,3,24),ini_len)
    #     pred_arimax(model, t, v, ini_len, i)


#### Other helper functions #################
    # plot_cat(all_d)
    


