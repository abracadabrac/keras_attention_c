from keras.layers import Dense, Input, Conv2D, merge, MaxPooling2D, Reshape, RepeatVector, Flatten, Permute, LSTM, Bidirectional, Activation, activations
from keras.models import Model
from data.reader import Data
from models.custom_recurrents import AttentionDecoder
import numpy as np

from data.vars import Vars

V = Vars()



def attention_network(data):
    """
    this function return a simple attention network with convolutional and LSTM layers
    It is a light network designed for the Hamelin word dataset
    :param data: element of the class Data defined in ./data/reader
    :return: neural network
    """
    i_ = Input(name='input', shape=(data.im_length, data.im_height, 1))

    p = {  # parameters
        "cc1": 4,  # number of convolution channels 1
        "kmp1": (2, 1),  # kernel max pooling 1
        "cc2": 16,  # ...
        "kmp2": (2, 2),
        "cc3": 32,
        "kmp3": (3, 2),
        "cc4": 64,
        "kmp4": (1, 1),
        "da": 50,  # attention dimension, internal representation of the attention cell
        "ld": 64 * 2 * 2,
        "do": data.vocab_size
    }
    total_maxpool_kernel = np.product([[p[k][0], p[k][1]] for k in p.keys() if k[:3] == "kmp"], axis=0)

    # Convolutions ##
    conv1_ = Conv2D(p["cc1"], (3, 3), padding="same", strides=p["kmp1"])(i_)
    conv1_ = Activation("relu")(conv1_)
    conv2_ = Conv2D(p["cc2"], (3, 3), padding="same", strides=p["kmp2"])(conv1_)
    conv2_ = Activation("relu")(conv2_)
    conv3_ = Conv2D(p["cc3"], (3, 3), padding="same", strides=p["kmp3"])(conv2_)
    conv3_ = Activation("relu")(conv3_)
    conv4_ = Conv2D(p["cc4"], (3, 3), padding="same", strides=p["kmp4"])(conv3_)
    conv4_ = Activation("relu")(conv4_)

    shape1 = (int(data.im_length / total_maxpool_kernel[0]),
              int(data.im_height / total_maxpool_kernel[1]) * p["cc4"])

    r_ = Reshape(shape1, name="collapse")(conv4_)

    lstm0_ = Bidirectional(LSTM(p["ld"], return_sequences=True), name="LSTM0")(r_)

    y_ = (AttentionDecoder(p["da"], p["do"], name='attention_' + str(p['da']))(lstm0_))

    return Model(inputs=i_, outputs=y_)



if __name__ == "__main__":

    data = Data(V.images_test_dir, V.labels_test_txt)
    model = attention_network(data)
    model.summary()
    json = model.to_json()

    print('fin')
