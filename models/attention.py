# soft attention 使用全连接层和sigmoid函数来进行相似性计算
from keras.layers import Dense, LSTM, Multiply, Permute, Flatten, RepeatVector, Lambda
import keras.layers.core
from keras import backend as K
from keras.models import Model
import tensorflow as tf
# import os

def attention_3d_block2(inputs, single_attention_vector=False):
    # 如果上一层是LSTM，需要return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    # 输入网络的数据的shape是(time_steps, input_dim)，这是方便输入到LSTM层里的输入格式。
    # 无论注意力层放在LSTM的前面还是后面，最终输入到注意力层的数据shape仍为(time_steps, input_dim)，
    # 对于注意力结构里的Dense层而言，(input_dim, time_steps)才是符合的，因此要进行维度变换。
    a = Permute((2, 1), name='attention_vec')(inputs)
    # 注意力结构里的Dense层，用于计算每个特征的权重。
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        # Ture: 多维特征共享一个注意力权重，False:每一维特征单独有一个注意力权重,即注意力权重也变成多维的了。
        # 取均值
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    # 到这步就已经是算好的注意力权重了，我们知道Attention的第二个结构就是乘法，而这个乘法要对应元素相乘，因此要再次对维度进行变换。
    # a:(None, 50,1) -> a_probs(None, 1, 50)
    a_probs = Permute((2, 1))(a)
    # 乘上了attention权重，但是并没有求和，好像影响不大
    # 如果分类任务，进行Flatten展开就可以了
    # element-wise
    # 将输入与注意力权重对应相乘
    output_attention_mul = Multiply()([inputs, a_probs])
    print("inputs", K.int_shape(output_attention_mul))
    print("a_probs", K.int_shape(a_probs))

    # os._exit_()
    return output_attention_mul


def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        # all layer outputs
        outputs = [
            layer.output for layer in model.layers if layer.name == layer_name]
    funcs = [K.function([inp] + [K.learning_phase()], [out])
             for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print('shape为', layer_activations.shape)
            print(layer_activations)
    return activations
