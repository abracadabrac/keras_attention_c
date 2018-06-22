from models.ANN import attention_network_1, standardlstm_network
from data.reader import Data
from data.vars import Vars
from loader import save_xp, load_xp_model
from utils.CER import CER


import datetime
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam

import numpy as np
import csv
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


def test_model(net, name):
    print(' ------ testing ------')
    os.makedirs('experiments/' + name + '/Test/')

    data = Data(V.images_test_dir, V.labels_test_txt)
    images, labels = data.generator(4450).__next__()
    decoded_label = data.decode_labels(labels, depad=True)

    prediction = net.predict(images)
    argmax_prediction = data.pred2OneHot(prediction)
    decoded_prediction = data.decode_labels(argmax_prediction, depad=True)

    with open('experiments/' + name + '/Test/predictions.csv', 'w') as f:
        fieldnames = ['label', 'prediction']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        for l, p in zip(decoded_label, decoded_prediction):
            writer.writerow({'label': l, 'prediction': p})

    cross_val = net.evaluate(images, labels, verbose=False)
    label_error = [CER(l, p) for l, p in zip(decoded_label, decoded_prediction)]
    label_error = np.sum(label_error) / len(label_error)

    with open('experiments/' + name + '/Test/loss.csv', 'w') as f:
        fieldnames = ['name', 'value']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'name': 'cross-entropie', 'value': cross_val})
        writer.writerow({'name': 'label error', 'value': label_error})


def main_training():

    data = Data(V.images_train_dir, V.labels_train_txt)

    validation_set = Data(V.images_valid_dir, V.labels_valid_txt)
    validation_data = validation_set.generator(4000).__next__()  # (x_val, y_val)


    net = attention_network_1(data)
    now = datetime.datetime.now().replace(microsecond=0)
    name = datetime.date.today().isoformat() + '/' + now.strftime("%H-%M-%S")

    os.makedirs("./experiments/" + name + '/weights/')

    comment = input("Enter (or not) a comment: ")

    with open("./experiments/" + name + "/comment.txt", "w") as f:
        f.write('   # init xp')
        f.write(comment)

    train_model(net, data, name,
                validation_data=validation_data,
                learning_rate=0.001,
                loss='categorical_crossentropy',
                batch_size=8,
                epoch=70,
                steps_per_epoch=1638)

    test_model(net, name)




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    named_args = parser.add_argument_group('named arguments')

    named_args.add_argument('-n', '--name', metavar='|',
                            help="""name of the experiment""",
                            required=False, default=None, type=str)

    args = parser.parse_args()

    main_training()
