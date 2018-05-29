from models.ANN import attention_network_1
from data.reader import Data
from data.vars import Vars

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import model_from_json

import models

import json
import os

V = Vars()


def train_model(net, data, name,
                validation_data,
                learning_rate=0.001,
                loss='mean_squared_error',
                batch_size=1,
                epoch=1,
                steps_per_epoch=1):
    tb = TensorBoard(log_dir='./experiments/' + name + '/TensorBoard/', histogram_freq=0, write_graph=True,
                     write_images=True)
    cp = ModelCheckpoint(filepath="./experiments/" + name + '/weights/w.{epoch:02d}-{val_loss:.2f}.hdf5')

    net.compile(optimizer=Adam(lr=learning_rate), loss=loss)

    net.fit_generator(data.generator(batch_size),
                      epochs=epoch,
                      validation_data=validation_data,
                      steps_per_epoch=steps_per_epoch,
                      callbacks=[tb, cp])

    save_experiment(net, name, learning_rate, loss, epoch, steps_per_epoch)


def mkexpdir():
    while True:
        try:
            name = input("Enter an experiment name: ")
            os.makedirs("./experiments/" + name + '/weights/')
            break
        except FileExistsError:
            print('Warning : experiment name %s already used' % name)

    return name


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
    net = model_from_json(net_json, custom_objects={'AttentionDecoder': models.custom_recurrents.AttentionDecoder})

    with open(d + '/meta_parameters.json', 'r') as f:
        meta_parameters = json.load(f)
    learning_rate = meta_parameters['learning_rate']
    loss = meta_parameters['loss']
    net.compile(optimizer=Adam(lr=learning_rate), loss=loss)

    net.load_weights(d + "/weights.h5")

    return net


# __________________________________________________ #


def evaluate_model(name, data_test):
    """
    Function to calculate the loss of the net on the data_test
    return: the loss
    """
    net = load_xp_model(name)

    X, Y = data_test.generator(1000).__next__()

    return net.evaluate(X, Y)


# __________________________________________________ #

def main_train():
    name = mkexpdir()  # in all the file 'name' implicitly refers to the name of an experiment

    data = Data(V.images_train_dir, V.labels_train_txt)

    validation_set = Data(V.images_valid_dir, V.labels_valid_txt)
    validation_data = validation_set.generator(1000).__next__()  # (x_val, y_val)

    net = attention_network_1(data)

    nb_data = len(data.labels_dict)  # total number of hand-written images in the train data-set

    train_model(net, data, name,
                validation_data=validation_data,
                learning_rate=0.001,
                loss='mean_squared_error',
                batch_size=8,
                epoch=50,
                steps_per_epoch=5)

    print('###----> training end <-----###')


def main_prediction():
    name = 'xp_2'  # in all the file 'name' implicitly refers to the name of an experiment

    data = Data(V.images_test_dir, V.labels_test_txt)

    net = load_xp_model(name)

    images, _ = data.generator(50).__next__()

    y = net.predict(images)
    #y = data.pred2OneHot(y)
    #y = data.decode_labels(y)

    return y

    print('###----> prediction end <-----###')


if __name__ == "__main__":
    main_prediction()
