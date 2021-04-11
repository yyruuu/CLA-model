import time
import matplotlib.dates as mdates
from keras.optimizers import Adam
import faulthandler
from data_processing import normalization, trian_test_split, inverse_trans, create_dataset
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from pandas import concat
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Bidirectional, Multiply, Permute, Flatten, MaxPooling1D, RepeatVector, Reshape, GRU, SimpleRNN
from attention import attention_3d_block2
from keras.models import Model
from keras import backend as K
import keras.layers.core
from sklearn.svm import SVR
import tensorflow as tf
mpl.use('TkAgg')
faulthandler.enable()

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


# 输入维度
INPUT_DIMS = 12
# 步长
TIME_STEPS = 1
# 训练集的大小
N_TRAIN_WEEKS = 187
# epoch
EPOCHS = 200
BATCH_SIZE = 24

# 1. 读取数据
data = pd.read_csv("../data/denoise.csv")

print("data shape", data.shape)
# data的第一列是index，第二列才是time
time_data = pd.read_excel("../data/time.xlsx").values
time_data = time_data[N_TRAIN_WEEKS+TIME_STEPS+1:]
time_data = time_data.reshape(len(time_data),)
values = data.iloc[:, 1:]
# 2. 归一化
scaled, scaler = normalization(values)

# 3. 数据格式转化及数据集划分
train_X, train_y, test_X, test_y = trian_test_split(
    scaled, N_TRAIN_WEEKS, TIME_STEPS)
train_X = train_X.reshape(train_X.shape[0], train_X.shape[2]*TIME_STEPS)
train_y = train_y.reshape(train_y.shape[0],)
test_X = test_X.reshape(test_X.shape[0], test_X.shape[2]*TIME_STEPS)
test_y = test_y.reshape(test_y.shape[0],)


mae_array = []
rmse_array = []
def main(test_X, test_y):
    m = SVR(kernel='rbf', gamma=0.01, C=100.0, degree=3)
    # 6. 预测
    m.fit(train_X, train_y)
    yhat = m.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], TIME_STEPS*INPUT_DIMS))
    inv_yhat = inverse_trans(yhat, test_X, scaler, N_TRAIN_WEEKS, INPUT_DIMS)
    inv_y = inverse_trans(test_y, test_X, scaler, N_TRAIN_WEEKS, INPUT_DIMS)

    # # 计算RMSE误差值
    mae_array.append(mean_absolute_error(inv_y, inv_yhat))
    rmse_array.append(sqrt(mean_squared_error(inv_y, inv_yhat)))


t0 = time.time()
for i in range(0, 20):
  main(test_X, test_y)
t1 = time.time() - t0

print("mean mae", np.mean(mae_array))
print("mean rmse", np.mean(rmse_array))
print("fit20次用时：", t1)

