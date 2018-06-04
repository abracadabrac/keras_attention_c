
from models.ANN import attention_network_1
from data.reader import Data

from keras import backend as K
from keras.layers import Input, Dense, Lambda, ThresholdedReLU
from keras.models import Model

from Run import load_xp_model
from data.reader import Vars

V = Vars()


def create_net():
    i_ = Input(name='input', shape=(16, 28))

    maxi_ = K.max(i_, axis=2)
    thri_ = ThresholdedReLU(theta=0.7)(maxi_)


    return Model(input=i_, output=thri_)


if __name__ == "__main__":
    name = 'xp_3'  # in all the file 'name' implicitly refers to the name of an experiment
    data = Data(V.images_test_dir, V.labels_test_txt)
    net = load_xp_model(name)
    images, labels = data.generator(50).__next__()
    x = net.predict(images)

    net_test = create_net()

    y = net_test.predict(x)

    print('fin du test')



