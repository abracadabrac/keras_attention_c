from keras.models import model_from_json
from keras.optimizers import Adam
import json

from models.custom_recurrents import AttentionDecoder

"""
implements functions to save and load models
"""


def save_xp(net, name, learning_rate, loss, epoch, steps_per_epoch):
    d = "./experiments/" + name

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



def load_xp_model(name, epoch=None):
    """
    :param epoch: the index of the training epoch at which loading the net
    :param name: name of the experiment
    :return: the network fully trained after the selected epoch, if epoch is None the function will charge the last one
    """
    d = 'experiments/' + name

    file = open(d + '/model.json', 'r')
    net_json = file.read()
    file.close()
    net = model_from_json(net_json, custom_objects={'AttentionDecoder': AttentionDecoder})

    with open(d + '/meta_parameters.json', 'r') as f:
        meta_parameters = json.load(f)
    learning_rate = meta_parameters['learning_rate']
    loss = meta_parameters['loss']
    net.compile(optimizer=Adam(lr=learning_rate), loss=loss)


    print(d + "/weights.h5")
    if epoch is None:
        print(d + "/weights.h5")
        net.load_weights(d + "/weights.h5")
    else:
        print("Warning loading epoch function unfinished")

    return net
