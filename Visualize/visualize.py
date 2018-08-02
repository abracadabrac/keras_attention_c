from loader import load_xp_model
from keras import Model
import json
import numpy as np
import cv2
from matplotlib.pyplot import imshow, show, figure
import scipy

import matplotlib.animation as animation

from models.custom_recurrents import AttentionDecoder
from data.reader import Data
from data.vars import Vars

import os

V = Vars(open('../../vars.json', 'r'))
os.chdir('..')


def create_net_attention_maps(net, name):
    """
    :param net:
    :param name:
    :return: the same net which outputs the attention maps instead of the labels
    """
    d = "./experiments/" + name

    with open(d + '/model.json', 'r') as f:
        params = json.load(f)

    layers = params['config']['layers']
    config_attention = layers[-1]['config']
    units_attention = config_attention['units']
    output_dim = config_attention['output_dim']
    last_layer_name = layers[-2]['name']

    i_ = net.input
    r_ = net.get_layer(last_layer_name).output
    a_ = (AttentionDecoder(units_attention, output_dim, name='attention_visualizer', return_probabilities=True)(r_))

    net = Model(inputs=i_, outputs=a_)
    net.load_weights(d + "/weights.h5")

    return net


def see_animation(name):
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


def maps(name):
    data = Data(V.images_train_dir, V.labels_train_txt)

    net = load_xp_model(name)
    net_attention = create_net_attention_maps(net, name)

    images, labels = data.generator(10).__next__()

    preds = data.decode_labels(data.pred2OneHot(net.predict(images)))
    atts = net_attention.predict(images)

    panel = np.zeros((35, 35, 3), dtype=np.int)
    att = atts[0, :, :, 0]

    x = 4
    y = 2
    panel[x:x + att.shape[0], y:y + att.shape[1], 1] = att * 255 / np.max(att)
    panel = cv2.resize(panel, (500, 500))

    cv2.imwrite('experiments/' + name + '/panel.jpg', panel)

    print('fin')


if __name__ == "__main__":
    name = "2018-06-29-15-46-30"
    maps(name)
