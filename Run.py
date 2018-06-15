from models.ANN import attention_network_1
from data.reader import Data
from data.vars import Vars
from utils.CER import CER
from loader import save_xp, load_xp_model

import datetime
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam

import os

V = Vars()


def train_model(net, data, name,
                validation_data,
                learning_rate=0.001,
                loss='categorical_crossentropy',
                batch_size=1,
                epoch=1,
                steps_per_epoch=1):

    tb = TensorBoard(log_dir='./experiments/' + name + '/TensorBoard/',
                     histogram_freq=1,
                     write_graph=True,
                     write_images=False)
    cp = ModelCheckpoint(filepath="./experiments/" + name + '/weights/w.{epoch:02d}-{val_loss:.2f}.hdf5')

    net.compile(optimizer=Adam(lr=learning_rate), loss=loss)

    save_xp(net, name, learning_rate, loss, epoch, steps_per_epoch)
    net.fit_generator(data.generator(batch_size),
                      epochs=epoch,
                      validation_data=validation_data,
                      steps_per_epoch=steps_per_epoch,
                      callbacks=[tb, cp])
    net.save_weights("./experiments/" + name + '/weights.h5')


def mkexpdir():

    now = datetime.datetime.now().replace(microsecond=0)
    name = datetime.date.today().isoformat() + '-' + datetime.time.isoformat(now.time())
    os.makedirs("./experiments/" + name + '/weights/')

    comment = input("Enter (or not) a comment: ")
    with open("./experiments/" + name + "/comment.txt", "w") as f:
        f.write('   # init xp')
        f.write(comment)

    return name


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

def main_training():
    name = mkexpdir()  # in all the file 'name' implicitly refers to the name of an experiment

    data = Data(V.images_train_dir, V.labels_train_txt)

    validation_set = Data(V.images_valid_dir, V.labels_valid_txt)
    validation_data = validation_set.generator(4000).__next__()  # (x_val, y_val)

    net = attention_network_1(data)

    nb_data = len(data.labels_dict)  # total number of hand-written images in the train data-set

    train_model(net, data, name,
                validation_data=validation_data,
                learning_rate=0.001,
                loss='categorical_crossentropy',
                batch_size=8,
                epoch=6,
                steps_per_epoch=1638)

    print('###----> training end <-----###')


def main_prediction():
    name = '2018-06-11-12:49:30'  # in all the file 'name' implicitly refers to the name of an experiment
    data = Data(V.images_test_dir, V.labels_test_txt)
    net = load_xp_model(name)
    images, labels = data.generator(50).__next__()
    decoded_label = data.decode_labels(labels, depad=True)

    prediction = net.predict(images)
    argmax_prediction = data.pred2OneHot(prediction)
    decoded_prediction = data.decode_labels(argmax_prediction, depad=True)

    i = 2
    print(decoded_prediction)
    print(decoded_label)
    print("error 1   ", net.evaluate(images, labels, verbose=False))

    print("WER", CER(decoded_prediction[i], decoded_label[i]))



    print('###----> prediction end <-----###')


if __name__ == "__main__":
    main_prediction()
