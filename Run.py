from models.ANN import attention_network_1
from data.reader import Data
from keras.callbacks import ModelCheckpoint, TensorBoard


def train_model(net, data, optimizer='rmsprop', loss='categorical_crossentropy'):
    tb = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    net.compile(optimizer=optimizer, loss=loss)

    net.fit_generator(data.generator(4),
                      epochs=1,
                      steps_per_epoch=300,
                      callbacks=[tb])

    net.save_weights("./weights/weights_model_1")


def random_predict(net, data):
    net.compile(optimizer='rmsprop', loss='catergorical_crossentropy')

    input = data.generator(2).__next__()[0]

    return net.predict(input)


if __name__ == "__main__":
    root = "/Users/charles/Data/Hamelin/"
    images_test_dir = root + "TST/test/"
    labels_test_txt = root + "test.txt"

    data = Data(images_test_dir, labels_test_txt)

    net = attention_network_1(data)
    print(net.summary())

    #train_model(net, data)

    print("fin")
