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
from keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Bidirectional, Multiply, Permute, Flatten, MaxPooling1D, RepeatVector, Reshape
from keras.models import Model
from keras import backend as K
import keras.layers.core
mpl.use('TkAgg')
faulthandler.enable()
def cnn_model(step, input_dim):
    """
    step: 步长
    input_dim：输入维度
    """
    inputs = Input(shape=(step, input_dim))
    # for pH
    x = Conv1D(filters=16, kernel_size=1, activation='elu')(inputs)  # , padding = 'same'
    x = Conv1D(filters=32, kernel_size=1, activation='elu')(x)  # , padding = 'same'
    
    # for NH3-N
    # x = Conv1D(filters=16, kernel_size=1, activation='elu')(inputs)  # , padding = 'same'
    # x = Conv1D(filters=32, kernel_size=1, activation='elu')(x)  # , padding = 'same'
    lstm_out = Dropout(0.2)(x)
    attention_mul = Flatten()(lstm_out)

    output = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model


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
time_data = pd.read_excel("../data/time.xlsx").values
time_data = time_data[N_TRAIN_WEEKS+TIME_STEPS+1:]
time_data = time_data.reshape(len(time_data),)
values = data.iloc[:, 1:]
# for NH3-N prediction
# values[['0', '3']] = values[['3', '0']]
# 2. 归一化
scaled, scaler = normalization(values)

# 3. 数据格式转化及数据集划分
train_X, train_y, test_X, test_y = trian_test_split(
    scaled, N_TRAIN_WEEKS, TIME_STEPS)


mae_array = []
rmse_array = []
def main(test_X, test_y):
    m = cnn_model(TIME_STEPS, INPUT_DIMS)
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    m.compile(optimizer=optimizer, loss='mae')
    history = m.fit([train_X], train_y, epochs=EPOCHS,
                    batch_size=BATCH_SIZE, validation_data=(test_X, test_y))
    # 6. 预测
    yhat = m.predict(test_X, verbose=0)
    print("rmse", sqrt(mean_squared_error(test_y, yhat)))
    print("mae", mean_absolute_error(test_y, yhat))

    test_X = test_X.reshape((test_X.shape[0], TIME_STEPS*INPUT_DIMS))
    inv_yhat = inverse_trans(yhat, test_X, scaler, N_TRAIN_WEEKS, INPUT_DIMS)
    inv_y = inverse_trans(test_y, test_X, scaler, N_TRAIN_WEEKS, INPUT_DIMS)

    mae_array.append(mean_absolute_error(inv_y, inv_yhat))
    rmse_array.append(sqrt(mean_squared_error(inv_y, inv_yhat)))


t0 = time.time()
for i in range(0, 20):
  main(test_X, test_y)
t1 = time.time() - t0

print("mean mae", np.mean(mae_array))
print("mean rmse", np.mean(rmse_array))
print("fit20次用时：", t1)
print(mae_array)

