from loader import load_xp_model
from keras import Model
import json

from models.custom_recurrents import AttentionDecoder
from data.reader import Data
from data.vars import Vars

import os

V = Vars()
os.chdir('..')

data = Data(V.images_test_dir, V.labels_test_txt)

def recreate_net(net, name):
    d = "./experiments/" + name

    with open(d + '/model.json', 'r') as f:
        params = json.load(f)

    params_attention = params['config']['layers'][-1]
    units = params_attention['units']
    output_dim = params_attention['output_dim']

    i_ = net.input
    r_ = net.get_layer("collapse").output
    a_ = (AttentionDecoder(units, output_dim, name='attention_visualizer')(r_))

    net = Model(inputs=i_,
                outputs=a_)

    net.load_weights(d + "/weights.h5")

    return net


def main_see_attention_maps():

    name = 'xp_5'
    net = load_xp_model(name)
    net_attention = recreate_net(net, name)
    print(net.summary())
    print(net_attention.summary())


if __name__ == '__main__':
    main_see_attention_maps()

