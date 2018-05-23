from models.ANN import attention_network_1
from data.reader import Data

from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import model_from_json

import json
import os


def train_model(net, data, name,
                learning_rate=0.001,
                loss='mean_squared_error',
                batch_size=1,
                epoch=1,
                steps_per_epoch=1):

    mkexpdir(name)            # make experiment directory

    tb = TensorBoard(log_dir='./experiments/' + name + '/TensorBoard/', histogram_freq=0, write_graph=True, write_images=True)

    net.compile(optimizer=Adam(lr=learning_rate), loss=loss)

    net.fit_generator(data.generator(batch_size),
                      epochs=epoch,
                      steps_per_epoch=steps_per_epoch,
                      callbacks=[tb])

    save_experiment(net, name, learning_rate, loss, epoch, steps_per_epoch)


def mkexpdir(name):
    try:
        os.makedirs("./experiments/" + name)
    except FileExistsError:
        print('Warning : experiment name %s already used' %name)


def save_experiment(net, name, learning_rate, loss, epoch, steps_per_epoch):
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

    net.save_weights(d + '/weights.h5')


def load_xp_model(name):
    """
    :param name: name of the experiment
    :return: the network fully trained after the last epoch
    """
    d = 'experiments/' + name

    file = open(d + '/model.json', 'r')
    net_json = file.read()
    file.close()
    net = model_from_json(net_json)

    with open('meta_parameters.json', 'r') as f:
        meta_parameters = json.load(f)
    learning_rate = meta_parameters['learning_rate']
    loss = meta_parameters['loss']
    net.compile(optimizer=Adam(lr=learning_rate), loss=loss)

    net.load_weights(d + "/weights.h5")

    return net

# __________________________________________________ #


def predict(net, data, nb_pred=50):
    images, labels = data.generator(nb_pred).__next__()

    return net.predict(images)


def evaluate_model(name, data_test):
    """
    Function to calculate the loss of the net on the data
    return: the loss
    """
    net = load_xp_model(name)

    X, Y = data_test.generator(1000).__next__()

    return net.evaluate(X, Y)


   # # # # # # # # # # #
# # diposable functions  # #
   # # # # # # # # # # #


def main_train():
    name = 'xp_1'     # in all the file 'name' implicitly refers to the name of an experiment
    root = "/Users/charles/Data/Hamelin/"       # dir containing TRAIN, TST and VAL
    images_train_dir = root + "TRAIN/train/"
    labels_train_txt = root + "train.txt"

    data = Data(images_train_dir, labels_train_txt)
    net = attention_network_1(data)

    nb_data = len(data.labels_dict)     # total number of hand-written images in the train data-set

    train_model(net, data, name,
                learning_rate=0.001,
                loss='mean_squared_error',
                batch_size=8,
                epoch=50,
                steps_per_epoch=nb_data)

    print('###----> training end <-----###')


def main_predict():
    name = 'xp_1'
    root = "/Users/charles/Data/Hamelin/"
    images_test_dir = root + "TST/test/"
    labels_test_txt = root + "test.txt"

    data = Data(images_test_dir, labels_test_txt)

    y = predict(name, data)

    print('fin evaluate')


if __name__ == "__main__":
    main_train()
