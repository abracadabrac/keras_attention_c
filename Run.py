from models.ANN import attention_network_1
from data.reader import Data
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam


def train_model(net, data, optimizer=Adam(lr=0.001), loss='mean_squared_error'):
    tb = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    net.compile(optimizer=optimizer, loss=loss)

    net.fit_generator(data.generator(4),
                      epochs=10,
                      steps_per_epoch=4453,
                      callbacks=[tb])

    net.save_weights("./weights/weights_model_1")


def predict(net, data, weights_hdf5=None):
    if weights_hdf5 is not None:
        net.load_weights(weights_hdf5)

    images, labels = data.generator(50).__next__()

    return net.predict(images)


def evaluate_model(net, data_test, weights_hdf5, optimizer=Adam(lr=0.001), loss='mean_squared_error'):
    """
    Function to calculate the loss of the net on the data
    return: the loss
    """
    net.load_weights(weights_hdf5)
    X, Y = data_test.generator(1000).__next__()
    net.compile(optimizer=optimizer, loss=loss)

    return net.evaluate(X, Y)


   # # # # # # # # # # #
# # diposable functions  # #
   # # # # # # # # # # #

def main1():
    root = "/Users/charles/Data/Hamelin/"
    images_test_dir = root + "VAL/valid/"
    labels_test_txt = root + "valid.txt"
    weights_hdf5 = "/Users/charles/Workspace/text_attention/weights/weights_model_1"

    data = Data(images_test_dir, labels_test_txt)
    net = attention_network_1(data)

    y = predict(net, data, weights_hdf5=weights_hdf5)

    print('fin main1')


if __name__ == "__main__":
    main1()
    print("fin")
