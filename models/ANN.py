from keras.layers import Dense, Input, Conv2D, merge, MaxPooling2D, Reshape, RepeatVector, Flatten, Permute, LSTM, Bidirectional
from keras.models import Model
from data.reader import Data
from models.custom_recurrents import AttentionDecoder
import numpy as np


# A model using the custom layer inspired by https://github.com/datalogue/keras-attention
def attention_network_1(data):
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
        "cc2": 8,  # ...
        "kmp2": (3, 2),
        "cc3": 16,
        "kmp3": (4, 2),
        "da": 128,  # attention dimension, internal representation of the attention cell
        "do": data.vocab_size  # dimension of the abstract representation the elements of the sequence
    }
    total_maxpool_kernel = np.product([[p[k][0], p[k][1]] for k in p.keys() if k[:3] == "kmp"], axis=0)

    # Convolutions ##
    c_1 = Conv2D(p["cc1"], (3, 3), padding="same")(i_)
    mp_1 = MaxPooling2D(pool_size=p["kmp1"])(c_1)
    c_2 = Conv2D(p["cc2"], (3, 3), padding="same")(mp_1)
    mp_2 = MaxPooling2D(pool_size=p["kmp2"])(c_2)
    c_3 = Conv2D(p["cc3"], (3, 3), padding="same")(mp_2)
    mp_3 = MaxPooling2D(pool_size=p["kmp3"])(c_3)

    shape_1 = (int(data.im_length / total_maxpool_kernel[0]),
               int(data.im_height / total_maxpool_kernel[1]) * p["cc3"])

    r_ = Reshape(shape_1)(mp_3)

    # Long Short Term Memory ##
    lstm_1 = Bidirectional(LSTM(16, return_sequences=True, dropout=0.1))(r_)
    lstm_2 = Bidirectional(LSTM(32, return_sequences=True, dropout=0.1))(lstm_1)

    y_ = (AttentionDecoder(p["da"], p["do"])(lstm_2))

    return Model(inputs=i_, outputs=y_)


if __name__ == "__main__":
    root = "/Users/charles/Data/Hamelin/"
    images_test_dir = root + "TST/test/"
    labels_test_txt = root + "test.txt"
    data = Data(images_test_dir, labels_test_txt)

    model = attention_network_1(data)
    model.summary()

    print('fin')
