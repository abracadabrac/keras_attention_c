from loader import load_xp_model
from keras import Model
import json
import numpy as np
from matplotlib.pyplot import imshow, show, figure

import matplotlib.animation as animation

from models.custom_recurrents import AttentionDecoder
from data.reader import Data
from data.vars import Vars

import os

V = Vars()
os.chdir('..')


def create_net_attention_maps(net, name):
    d = "./experiments/" + name

    with open(d + '/model.json', 'r') as f:
        params = json.load(f)

    params_attention = params['config']['layers'][-1]['config']
    units = params_attention['units']
    output_dim = params_attention['output_dim']

    i_ = net.input
    r_ = net.get_layer("collapse").output
    a_ = (AttentionDecoder(units, output_dim, name='attention_visualizer', return_probabilities=True)(r_))

    net = Model(inputs=i_, outputs=a_)
    net.load_weights(d + "/weights.h5")

    return net


def see_maps(name):
    data = Data(V.images_train_dir, V.labels_train_txt)

    net = load_xp_model(name)
    net_attention = create_net_attention_maps(net, name)

    images, labels = data.generator(1).__next__()

    preds = data.decode_labels(data.pred2OneHot(net.predict(images)))
    atts = net_attention.predict(images)

    im_index = 0
    image = images[im_index, :, :, 0]
    pred = preds[im_index]
    att = atts[im_index, :, :, 0]

    ratio = image.shape[0] / att.shape[0]
    list_images_att = []
    for a in att:
        image_att = np.copy(image)
        for x in range(image.shape[0]):
            image_att[x, :] = image_att[x, :] * a[int(x / ratio)]

        list_images_att.append([imshow(np.rot90(image_att, k=1), animated=True)])
    list_images_att.append([imshow(np.rot90(np.zeros((384, 28)), k=1), animated=True)])

    fig = figure()
    ani = animation.ArtistAnimation(fig, list_images_att, interval=800, blit=True, repeat_delay=6e5)
    show()


def main_annimation():
    name = '2018-06-18-18:29:09'

    see_maps(name)


def main_see_attention_maps():
    name = '2018-06-18-18:29:09'

    see_maps(name)


if __name__ == '__main__':
    main_see_attention_maps()
