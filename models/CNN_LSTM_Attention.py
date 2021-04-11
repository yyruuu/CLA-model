from keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Bidirectional, Multiply, Permute, Flatten, MaxPooling1D, RepeatVector, Reshape, GRU
from attention import attention_3d_block2
from keras.models import Model
from keras import backend as K
import keras.layers.core
import tensorflow as tf

def cnn_lstm_attention_model(step, input_dim):
    """
    step: 步长
    input_dim：输入维度
    """
    inputs = Input(shape=(step, input_dim))

    x = Conv1D(filters=64, kernel_size=1, activation='elu')(inputs)  # , padding = 'same'
    # x = Conv1D(filters=32, kernel_size=1, activation='elu')(x)  # , padding = 'same'
    print("Conv1D 111", K.int_shape(x))
    x = Dropout(0.3)(x)
    # x = Conv1D(filters=64, kernel_size=2, activation='elu')(x)  # , padding = 'same'
    # x = MaxPooling1D(2)(x)
    #lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    # 对于GPU可以使用CuDNNLSTM
    #lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_out = LSTM(64, return_sequences=True)(x)
    
    # lstm_out = LSTM(16, return_sequences=True)(x)
    
    # lstm_out = LSTM(50, activation='elu', return_sequences=True, dropout=0.2)(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    print("lstm out", K.int_shape(lstm_out))
    # CNN和LSTM已经对特征进行提取了，这时候再用注意力机制效果可能会更好
    attention_mul = attention_3d_block2(lstm_out)

    attention_mul = Flatten()(attention_mul)

    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model

# step 4  0.068
def cnn_lstm_step4(step, input_dim):
    """
    step: 步长
    input_dim：输入维度
    """
    inputs = Input(shape=(step, input_dim))

    # x = Conv1D(filters=32, kernel_size=1, activation='relu')(inputs)  # , padding = 'same'

    x = Conv1D(filters=64, kernel_size=1, activation='elu')(inputs)  # , padding = 'same'
    x = Conv1D(filters=32, kernel_size=1, activation='elu')(x)  # , padding = 'same' 

    print("Conv1D 111", K.int_shape(x))
    x = Dropout(0.3)(x)
    lstm_out = LSTM(64, return_sequences=True, activation='elu')(x)

    lstm_out = Dropout(0.2)(lstm_out)
    print("lstm out", K.int_shape(lstm_out))
    # CNN和LSTM已经对特征进行提取了，这时候再用注意力机制效果可能会更好
    attention_mul = attention_3d_block2(lstm_out)
    attention_mul = Flatten()(attention_mul)

    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model

def cnn_lstm_nh3n(step, input_dim):
    """
    step: 步长
    input_dim：输入维度
    """
    inputs = Input(shape=(step, input_dim))

    x = Conv1D(filters=32, kernel_size=1, activation='elu')(inputs)  # , padding = 'same'
    x = Conv1D(filters=64, kernel_size=1, activation='elu')(x)  # , padding = 'same'


    print("Conv1D 111", K.int_shape(x))
    lstm_out = LSTM(40, return_sequences=True, activation='elu')(x)
    lstm_out = Dropout(0.2)(lstm_out)
    print("lstm out", K.int_shape(lstm_out))
    # CNN和LSTM已经对特征进行提取了，这时候再用注意力机制效果可能会更好
    attention_mul = attention_3d_block2(lstm_out)
    attention_mul = Flatten()(attention_mul)

    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model


def cnn_lstm_ph(step, input_dim):
    """
    step: 步长
    input_dim：输入维度
    """
    inputs = Input(shape=(step, input_dim))
    x = Conv1D(filters=32, kernel_size=1, activation='elu')(inputs)  # , padding = 'same'
    x = Conv1D(filters=64, kernel_size=1, activation='elu')(x)  # , padding = 'same'

    print("Conv1D 111", K.int_shape(x))
    lstm_out = LSTM(36, return_sequences=True, activation='elu')(x)
    lstm_out = LSTM(24, return_sequences=True, activation='elu', dropout=0.2)(lstm_out)
    
    lstm_out = Dropout(0.2)(lstm_out)
    print("lstm out", K.int_shape(lstm_out))
    # CNN和LSTM已经对特征进行提取了，这时候再用注意力机制效果可能会更好
    # attention_mul = attention_3d_block2(lstm_out)
    attention_mul = Flatten()(lstm_out)
    # attention_mul = Flatten()(attention_mul)


    output = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model


def cnn_lstm_do(step, input_dim):
    """
    step: 步长
    input_dim：输入维度
    """
    inputs = Input(shape=(step, input_dim))
    x = Conv1D(filters=32, kernel_size=1, activation='elu')(inputs)  # , padding = 'same'
    x = Conv1D(filters=64, kernel_size=1, activation='elu')(x)  # , padding = 'same'
    lstm_out = LSTM(36, return_sequences=True, activation='elu')(x)
    lstm_out = LSTM(24, return_sequences=True, activation='elu')(x)
    lstm_out = Dropout(0.2)(lstm_out)
    print("lstm out", K.int_shape(lstm_out))
    # CNN和LSTM已经对特征进行提取了，这时候再用注意力机制效果可能会更好
    attention_mul = attention_3d_block2(lstm_out)
    attention_mul = Flatten()(attention_mul)


    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model


def cnn_lstm_cod(step, input_dim):
    """
    step: 步长
    input_dim：输入维度
    """
    inputs = Input(shape=(step, input_dim))

    x = Conv1D(filters=64, kernel_size=1, activation='elu')(inputs)  # , padding = 'same'

    print("Conv1D 111", K.int_shape(x))
    x = Dropout(0.3)(x)
    # 对于GPU可以使用CuDNNLSTM
    lstm_out = LSTM(32, return_sequences=True, activation='elu')(x)
    lstm_out = LSTM(32, return_sequences=True, dropout=0.2)(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    print("lstm out", K.int_shape(lstm_out))
    # CNN和LSTM已经对特征进行提取了，这时候再用注意力机制效果可能会更好
    attention_mul = attention_3d_block2(lstm_out)
    attention_mul = Flatten()(attention_mul)

    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model
    

#NH3-N model
def cnn_lstm_ph2(step, input_dim):
    """
    step: 步长
    input_dim：输入维度
    """
    inputs = Input(shape=(step, input_dim))
    x = Conv1D(filters=32, kernel_size=1, activation='elu')(inputs)  # , padding = 'same'
    x = Conv1D(filters=64, kernel_size=1, activation='elu')(x)  # , padding = 'same'
    lstm_out = LSTM(32, return_sequences=True, activation='elu')(x)
    lstm_out = LSTM(24, return_sequences=True, activation='elu', dropout=0.2)(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    print("lstm out", K.int_shape(lstm_out))
    # CNN和LSTM已经对特征进行提取了，这时候再用注意力机制效果可能会更好
    attention_mul = attention_3d_block2(lstm_out)
    attention_mul = Flatten()(attention_mul)


    output = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model

#pH model
def cnn_lstm_ph3(step, input_dim):
    """
    step: 步长
    input_dim：输入维度
    """
    inputs = Input(shape=(step, input_dim))
    x = Conv1D(filters=16, kernel_size=1, activation='elu')(inputs)  # , padding = 'same'
    x = Conv1D(filters=32, kernel_size=1, activation='elu')(x)  # , padding = 'same'
    lstm_out = LSTM(24, return_sequences=True, activation='elu')(x)
    lstm_out = LSTM(24, return_sequences=True, activation='elu', dropout=0.2)(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    print("lstm out", K.int_shape(lstm_out))
    # CNN和LSTM已经对特征进行提取了，这时候再用注意力机制效果可能会更好
    attention_mul = attention_3d_block2(lstm_out)
    attention_mul = Flatten()(attention_mul)


    output = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model


