from models.ANN import attention_network_1
from data.reader import Data
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
import os


def train_model(net, data, name,
                optimizer=Adam(lr=0.001),
                loss='mean_squared_error',
                batch_size=1,
                epoch=1,
                steps_per_epoch=1):

    mkexpdir(name)            # make experiment directory

    tb = TensorBoard(log_dir='./experiments/' + name + 'TensorBoard/', histogram_freq=0, write_graph=True, write_images=True)

    net.compile(optimizer=optimizer, loss=loss)

    net.fit_generator(data.generator(batch_size),
                      epochs=epoch,
                      steps_per_epoch=steps_per_epoch,
                      callbacks=[tb])

    save_experiment(net, name)


def mkexpdir(name):
    files = os.listdir("./experiments")
    try:
        assert name not in files
    except AssertionError:
        print('Warning : experiment name %s already used' %name)

    os.makedirs("./experiments/" + name)


def save_experiment(net, name):
    d = "./experiments/" + name

    with open(d + "model.json", "w") as json_file:
        json_file.write(net.to_json())

    with open(d + 'summary.txt', 'w') as fh:
        net.summary(print_fn=lambda x: fh.write(x + '\n'))

    net.save_weights(d + '/weights.h5')


# __________________________________________________ #

def predict(net, data, weights_hdf5=None):
    images, labels = data.generator(50).__next__()

    return net.predict(images)


def evaluate_model(net, data_test, weights_hdf5, optimizer=Adam(lr=0.001), loss='mean_squared_error'):
    """
    Function to calculate the loss of the net on the data
    return: the loss
    """
    if weights_hdf5 is not None:
        net.load_weights(weights_hdf5)

    X, Y = data_test.generator(1000).__next__()
    net.compile(optimizer=optimizer, loss=loss)

    return net.evaluate(X, Y)


   # # # # # # # # # # #
# # diposable functions  # #
   # # # # # # # # # # #


def main_train():
    name = 'xp_1'
    root = "/Users/charles/Data/Hamelin/"
    images_train_dir = root + "TRAIN/train/"
    labels_train_txt = root + "train.txt"

    data = Data(images_train_dir, labels_train_txt)
    net = attention_network_1(data)

    train_model(net, data, name)

    print('fin train')


def main_evaluate():
    name = ' '
    root = "/Users/charles/Data/Hamelin/"
    images_test_dir = root + "TST/test/"
    labels_test_txt = root + "test.txt"
    weights_hdf5 = "./weights/" + name

    data = Data(images_test_dir, labels_test_txt)
    net = attention_network_1(data)

    y = predict(net, data, weights_hdf5=weights_hdf5)

    print('fin evaluate')


if __name__ == "__main__":
    main_train()
