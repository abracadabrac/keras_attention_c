
from keras.layers import TimeDistributed, Activation, Dense, Input, Bidirectional, Conv2D, Flatten, MaxPooling2D, RNN
from keras.layers import SimpleRNNCell
from keras.models import Model
from data.reader import Data
from models.custom_recurrents import AttentionDecoder



def attention_network_inter_layers(D):
    i_ = Input(name='input', shape=(D.im_length, D.im_height, 1))

    c_1 = Conv2D(8, (3, 3), padding="same", name="conv_1")(i_)
    c_2 = Conv2D(16, (3, 3), padding="same", name="conv_2")(c_1)

    return Model(inputs=i_, outputs=[c_1, c_2])


def recurrent_4DimInput(D):
    i_ = Input(name='input', shape=(D.im_length, D.im_height, 1))

    r_ = Bidirectional(RNN(SimpleRNNCell(32), name="recurrent"))(i_)

    return Model(inputs=i_, outputs=r_)


if __name__ == "__main__":
    data = Data()
    model = recurrent_4DimInput(data)
    print(model.summary())

