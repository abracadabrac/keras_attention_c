from keras.models import model_from_json
from keras.optimizers import Adam
import json
import os

from models.custom_recurrents import AttentionDecoder

from data.vars import Vars

V = Vars()

"""
implements functions to save and load models
"""


def save_xp(net, name, learning_rate, loss, epoch, steps_per_epoch):
    d = V.experiments_folder + "/keras/" + name

    meta_parameters = {'learning_rate': learning_rate,
                       'loss': loss,
                       'epoch': epoch,
                       'steps_per_epoch': steps_per_epoch}
    with open(d + '/meta_parameters.json', 'w') as f:
        json.dump(meta_parameters, f)

    with open(d + "/model.json", "w") as f:
        f.write(net.to_json())

    with open(d + '/architecture_summary.txt', 'w') as f:
        net.summary(print_fn=lambda x: f.write(x + '\n'))



def load_xp_model(name):
    """
    :param epoch: the index of the training epoch at which loading the net
    :param name: name of the experiment
    :return: the network fully trained after the selected epoch, if epoch is None the function will charge the last one
    """
    d = V.experiments_folder + "/keras/" + name

    file = open(d + '/model.json', 'r')
    net_json = file.read()
    file.close()
    net = model_from_json(net_json, custom_objects={'AttentionDecoder': AttentionDecoder})

    with open(d + '/meta_parameters.json', 'r') as f:
        meta_parameters = json.load(f)
    learning_rate = meta_parameters['learning_rate']
    loss = meta_parameters['loss']
    net.compile(optimizer=Adam(lr=learning_rate), loss=loss)

    weights = os.listdir(d + '/weights')
    weights.sort()

    net.load_weights(d + '/weights/' + weights[-1])

    return net


if __name__ == '__main__':
    net = load_xp_model('2018-07-10-16-37-16')
