from keras.layers import Dense, Input, Conv2D, merge, MaxPooling2D, Reshape, RepeatVector, Flatten, Permute
from keras.models import Model
from data.reader import Data
from models.custom_recurrents import AttentionDecoder
import numpy as np


# A model using the custom layer inspired by https://github.com/datalogue/keras-attention
def attention_network_1(data):
    i_ = Input(name='input', shape=(data.im_length, data.im_height, 1))

    p = {  # parameters
        "kc1": 4,  # kernel convolution 1
        "kmp1": (2, 2),  # kernel max pooling 1
        "kc2": 8,  # ...
        "kmp2": (2, 2),
        "kc3": 16,
        "kmp3": (6, 1),
        "da": 16,  # attention dimension, internal representation of the attention cell
        "do": 30  # output dimension
    }
    total_maxpool_kernel = np.product([[p[k][0], p[k][1]] for k in p.keys() if k[:3] == "kmp"], axis=0)
    output_shape = (int(data.im_length / total_maxpool_kernel[0]),
                    int(data.im_height / total_maxpool_kernel[1]) * p["kc3"])

    c_1 = Conv2D(p["kc1"], (3, 3), padding="same", name="conv_1")(i_)
    mp_1 = MaxPooling2D(pool_size=p["kmp1"])(c_1)
    c_2 = Conv2D(p["kc2"], (3, 3), padding="same", name="conv_2")(mp_1)
    mp_2 = MaxPooling2D(pool_size=p["kmp2"])(c_2)
    c_3 = Conv2D(p["kc3"], (3, 3), padding="same", name="conv_3")(mp_2)
    mp_3 = MaxPooling2D(pool_size=p["kmp3"])(c_3)
    a_ = Reshape(output_shape)(mp_3)

    y_ = (AttentionDecoder(p["da"], p["do"])(a_))

    return Model(inputs=i_, outputs=y_)


if __name__ == "__main__":
    root = "/Users/charles/Data/Hamelin/"
    images_test_dir = root + "TST/test/"
    labels_test_txt = root + "test.txt"
    data = Data(images_test_dir, labels_test_txt)

    model = attention_network_1(data)
    model.summary()

    print('fin')
