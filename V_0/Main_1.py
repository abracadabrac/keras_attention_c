import numpy as np
from random import randint

from keras.models import Sequential
from keras.layers import Input, LSTM
from keras.models import Model
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import RepeatVector

from attention_decoder import AttentionDecoder

n_features = 3
width = 50
height = 50


def create_network(nb_attention_cells=1):
    model = Sequential()
    model.add(AttentionDecoder(nb_attention_cells, n_features, input_shape=(width, height)))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model


if __name__ == "__main__":
    model = create_network(10)

    model.summary()
