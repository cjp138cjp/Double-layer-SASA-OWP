from vmdpy import VMD
import pandas as pd
import numpy as np
import csv
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import explained_variance_score
from sklearn import metrics
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn import metrics
from datetime import datetime
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
from pyhht.emd import EMD
from pyhht.visualization import plot_imfs
from sklearn.metrics import mean_absolute_error # 平方绝对误差
# vmd分解 #交叉验证解决data leakage
df = pd.read_csv(r".\data\1.csv")
# 标准化
df_list = pd.concat([df['kw2'], df['kw3'], df['kw4']], ignore_index=True).dropna()
df_list_max = df_list.max()
df_list_min = df_list.min()
df = (df_list-df_list_min)/(df_list_max-df_list_min)
# 最大最小标准化
time_series = np.array(df)
K = 18
alpha = 3000  # moderate bandwidth constraint   2000
tau = 0.  # noise-tolerance (no strict fidelity enforcement)
DC = 0  # no DC part imposed
init = 1  # initialize omegas uniformly
tol = 1e-7
imfs, _, _ = VMD(time_series, alpha, tau, K, DC, init, tol)
print(imfs,imfs.shape)
# 固定随机种子以复现结果
seed = 42
np.random.seed(seed)
lag_days = 96*3

from scipy.fftpack import fft
import warnings
warnings.filterwarnings("ignore")
from vmdpy import VMD

yuce_len = 960

predictions1d1, predictions1d2, predictions1d3, predictions1d4, predictions1d5, predictions1d6, predictions1d7, predictions1d8, predictions1d9, predictions1d10, predictions1d11, predictions1d12, predictions1d13, predictions1d14, predictions1d15, predictions1d16, predictions1d17, predictions1d18 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
predictions1d_list = [predictions1d1, predictions1d2, predictions1d3, predictions1d4, predictions1d5, predictions1d6,
                      predictions1d7, predictions1d8, predictions1d9, predictions1d10, predictions1d11, predictions1d12,
                      predictions1d13, predictions1d14, predictions1d15, predictions1d16, predictions1d17,
                      predictions1d18]


def timeseries_to_supervised(data, lag):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

import tensorflow as tf
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    strategy = tf.distribute.OneDeviceStrategy("GPU:0")
    with strategy.scope():
        model = Sequential()
        model.add(LSTM(neurons, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=2, shuffle=False)
    return model


# make a one-step forecast
def forecast_lstm(model, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X)
    return yhat[0, 0]


from numpy.random import seed

seed(5)
import tensorflow

tensorflow.random.set_seed(5)


emdlstm = [0] * yuce_len
for c in range(K):

    test = imfs[c][-yuce_len:int(len(time_series))]
    c2d = []
    for i in imfs[c]:
        c2d.append([i, i])
    print(np.array(c2d).shape)
    scaler = StandardScaler()  # 标准化转换
    scaler.fit(c2d)  # 训练标准化对象
    supervised = scaler.transform(c2d)
    print(np.array(supervised).shape)  # 转换数据集
    c1d = []
    for j in supervised:
        c1d.append(j[0])
    supervised = timeseries_to_supervised(c1d, lag_days)
    print(np.array(supervised).shape)
    train_scaled, test_scaled = supervised[0:-yuce_len], supervised[-yuce_len:int(
        len(supervised))]
    print(np.array(train_scaled).shape, np.array(test_scaled).shape)
    train_scaled = np.array(train_scaled)
    test_scaled = np.array(test_scaled)

    print("开始")
    # fit the model
    lstm_model = fit_lstm(train_scaled, 128, 5, 27)
    #     # forecast the entire training dataset to build up state for forecasting
    #     train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    #     lstm_model.predict(train_reshaped, batch_size=1)

    # walk-forward validation on the test data
    predictions = list()
    testdata = np.array(test_scaled[:, 0:-1]).reshape(np.array(test_scaled[:, 0:-1]).shape[0], 1,
                                                      np.array(test_scaled[:, 0:-1]).shape[1])
    predictions = lstm_model.predict(testdata)
    predictions = np.array(predictions).reshape(1, len(predictions))[0]
    print(predictions, len(predictions))
    #     for i in range(len(test_scaled)):
    #       # make one-step forecast
    #         X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    #         yhat = forecast_lstm(lstm_model, X)
    #       # store forecast
    #         predictions.append(yhat)
    print("结束")

    predictions2d = []
    for i in predictions:
        predictions2d.append([i, i])
    predictions2dsupervised = scaler.transform(c2d)  # 转换数据集

    predictions2d = scaler.inverse_transform(predictions2d)
    predictions1d_i = predictions1d_list[c]
    for j in predictions2d:
        predictions1d_i.append(j[0])

    # report performanceprint("MSE:",mean_sq2uared_error(test,predictions1d))

    print("R2 = ", metrics.r2_score(test, predictions1d_i))  # R2
    # line plot of observed vs predicted

    predictions1d_list[c] = np.array(predictions1d_i)

    emdlstm = list(np.add(emdlstm, predictions1d_i))
    #     emdlstm += predictions1d_i
    print(len(emdlstm))
#     predictions1d1+predictions1d2+predictions1d3+predictions1d4+predictions1d5+predictions1d6+predictions1d7+predictions1d8+predictions1d9+predictions1d10+predictions1d11+predictions1d12+predictions1d13+predictions1d13+predictions1d14+predictions1d15+predictions1d16+predictions1d17+predictions1d18
# print(np.array(emdlstm).shape)
print(np.array(emdlstm).shape)
from numpy import concatenate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

testtest = time_series[-yuce_len:]
print(len(testtest))

def inverse_transform_col1(y,df_max,df_min):
    '''scaler是对包含多个feature的X拟合的,y对应其中一个feature,n_col为y在X中对应的列编号.返回y的反归一化结果'''
    y = y.copy()
    y = np.array(y)
    y *= (df_max-df_min)
    y += df_min
    return y

testtest = inverse_transform_col1(testtest,df_list_max,df_list_min)
emdlstm = inverse_transform_col1(emdlstm,df_list_max,df_list_min)
print(len(testtest),len(emdlstm))



dt1 = pd.DataFrame({"true":np.array(testtest).reshape(-1,),"predic":np.array(emdlstm).reshape(-1,)})
dt1.to_excel(r".\vmd-lstm.xlsx")

print("MSE:", mean_squared_error(testtest, emdlstm))
print("RMSE:", sqrt(mean_squared_error(testtest, emdlstm)))
print("MAE = ", mean_absolute_error(testtest, emdlstm))  # R2
print("R2 = ", metrics.r2_score(testtest, emdlstm))  # R2

# line plot of observed vs predicted
fig = pyplot.gcf()
fig.set_size_inches(18.5, 10.5)
pyplot.plot(testtest)
pyplot.plot(emdlstm)
pyplot.legend(['True', 'R1'])
pyplot.show()
