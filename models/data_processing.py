import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat


def create_dataset(dataset, look_back):
    '''
    将时序数据转换为监督学习的格式，滑动窗口
    dataset：数据集
    look_back：步长
    '''
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)
    return TrainX, Train_Y


def create_dataset2(data, n_predictions, n_next):
    '''
    将时序数据转换为监督学习的格式，滑动窗口
    data：数据集
    n_predictions：step
    n_next： 预测未来多久的数据
    '''
    dim = data.shape[1]
    train_X, train_Y = [], []
    for i in range(data.shape[0]-n_predictions-n_next-1):
        a = data[i:(i+n_predictions), :]
        train_X.append(a)
        tempb = data[(i+n_predictions):(i+n_predictions+n_next), :]
        b = []
        for j in range(len(tempb)):
            for k in range(dim):
                b.append(tempb[j, k])
        train_Y.append(b)
    train_X = np.array(train_X, dtype='float64')
    train_Y = np.array(train_Y, dtype='float64')

    return train_X, train_Y


def normalization(data):
    """
    最大最小值归一化
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    return scaled, scaler


def inverse_trans(predict, origin_data, scaler, n_weeks, n_features):
    # 将预测列据和后11列数据拼接，因后续逆缩放时，数据形状要符合 n行*12列 的要求
    inv_predict = np.column_stack((predict, origin_data[:, -11:]))
    # 对拼接好的数据进行逆缩放
    inv_predict = scaler.inverse_transform(inv_predict)
    inv_predict = inv_predict[:, 0]
    return inv_predict


def trian_test_split(data, train_weeks, step):
    """
    data: 传入归一化后的数据集
    train_weeks：前多少数据用于训练
    step: 步长
    """
    values = data
    y_data = values[:, 0].reshape(len(values), 1)

    # 构造数据集
    # X, _ = create_dataset(values, step)
    # _, Y = create_dataset(y_data, step)
    X, _ = create_dataset2(values, step, 1)
    # X, _ = create_dataset(values, step)
    # _, Y = create_dataset(y_data, step)
    _, Y = create_dataset2(y_data, step, 1)

    print("y datas", y_data)
    print("YYYY", Y)

    # 划分为训练集和测试集
    train_X = X[:train_weeks, :]
    train_y = Y[:train_weeks, :]

    test_X = X[train_weeks:, :]
    test_y = Y[train_weeks:, :]
    print("训练X 训练y 测试X 测试y的shape", train_X.shape,
          train_y.shape, test_X.shape, test_y.shape)
    return train_X, train_y, test_X, test_y


# 来源于《Python深度学习》
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][0]
        print("samples", len(samples), samples)
        print("targets", len(targets), targets)
        yield samples, targets


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
            data: Sequence of observations as a list or NumPy array.
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
            Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def dataform(input_dim, scaled_data, step, n_train_weeks, n=0):
  """
  input_dim: 输入维度
  scaled_data： 归一化后的数据
  step：步长
  n_train_weeks： 训练周数
  n=0:ph，n=3:nh3-n
  """
  n_features = input_dim
  reframed = series_to_supervised(scaled_data, step, 1, True)

  values = reframed.values
  # 前三年半的数据来训练
  train = values[:n_train_weeks, :]
  test = values[n_train_weeks:, :]
  n_obs = step * n_features
  train_X, train_y = train[:, :n_obs], train[:, -n_features+n]
  test_X, test_y = test[:, :n_obs], test[:, -n_features+n]
  return train_X, train_y, test_X, test_y
