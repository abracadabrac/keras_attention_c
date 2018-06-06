from loader import load_xp_model
from keras import Model

from models.custom_recurrents import AttentionDecoder
from data.reader import Data
from data.vars import Vars

import os

V = Vars()
os.chdir('..')

data = Data(V.images_test_dir, V.labels_test_txt)

p = {  # parameters
    "cc1": 4,  # number of convolution channels 1
    "kmp1": (2, 1),  # kernel max pooling 1
    "cc2": 16,  # ...
    "kmp2": (3, 2),
    "cc3": 64,
    "kmp3": (4, 2),
    "da": 200,  # attention dimension, internal representation of the attention cell
    "do": data.vocab_size  # dimension of the abstract representation the elements of the sequence
}


def recreate_net(net):
    i_ = net.input
    r_ = net.get_layer("collapse").output
    a_ = (AttentionDecoder(p["da"], p["do"], name='attention_visualizer')(r_))

    net = Model(inputs=i_,
                outputs=a_)

    net.load_weights(d + "/weights.h5")

    return net


def main_see_attention_maps():

    name = 'xp_5'
    net = load_xp_model(name)
    net_attention = recreate_net(net)
    print(net.summary())
    print(net_attention.summary())


if __name__ == '__main__':
    main_see_attention_maps()

