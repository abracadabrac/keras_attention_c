from keras.layers import Dense, Input,  Conv2D, merge, MaxPooling2D, Reshape, RepeatVector, Flatten, Permute
from keras.models import Model
from data.reader import Data
from models.custom_recurrents import AttentionDecoder
import numpy as np

# A model using the custom layer inspired by https://github.com/datalogue/keras-attention
def attention_network_1(data):
    i_ = Input(name='input', shape=(data.im_length, data.im_height, 1))

    p = {                   # parameters
        "dc1" : 8,          # dimension_convolution_1
        "dc2" : 8,          # ...
        "da" : 16,          # attention dimension, internal representation of the attention cell
        "do" : 30           # output dimension
    }

    c_1 = Conv2D(p["dc1"], (3, 3), padding="same", name="conv_1")(i_)
    mp_1 = MaxPooling2D(pool_size=(1, 2))(c_1)
    c_2 = Conv2D(p["dc2"], (3, 3), padding="same", name="conv_2")(mp_1)
    mp_2 = MaxPooling2D(pool_size=(1, 4))(c_2)
    a_ = Reshape((data.im_length, int(data.im_height * p["dc2"] / 8)))(mp_2)

    y_ = (AttentionDecoder(p["da"], p["do"])(a_))

    return Model(inputs=i_, outputs=y_)

if __name__ == "__main__":
    data = Data()
    model = attention_network_1(data)
    model.summary()

    print('fin')