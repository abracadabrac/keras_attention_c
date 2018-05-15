import numpy as np
import random
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class RData:

    def __init__(self, im_length=64, im_height=64, nb_data=32):  # random init

        self.im_length = im_length
        self.im_height = im_height
        self.nb_data = nb_data
        self.truths = np.array([[[random.randint(0, 1)
                                  for _ in range(30)]
                                 for _ in range(self.im_length)]
                                for _ in range(self.nb_data)])

        self.images = np.array([10 * np.random.rand(self.im_length, self.im_height) for _ in range(self.nb_data)])

    def generator(self, batch_size):
        instance_id = range(self.nb_data)
        while True:
            try:
                batch_ids = random.sample(instance_id, batch_size)
                yield (
                    np.array(self.images[batch_ids], dtype=int).reshape(batch_size, self.im_length, self.im_height, 1),
                    np.array(self.truths[batch_ids]))
            except Exception as e:
                print('EXCEPTION OMG')
                print(e)
                yield None, None


def get_labels_dict(labels_txt):
    labels_dict = {}
    txt = open(labels_txt).read()  # text contenant les labels et les noms des fichiers images
    n_index = [i for i, x in enumerate(txt) if x == "\n"]  # indices des retours a la ligne das le text

    line = txt[0:n_index[0]]

    for i in range(len(n_index) - 1):
        s = line.index(' ')
        t = line.index('/')
        labels_dict[line[t + 1:s]] = line[s + 1:]

        d = n_index[i] + 1
        f = n_index[i + 1]
        line = txt[d:f]

    s = line.index(' ')
    t = line.index('/')
    labels_dict[line[t + 1:s]] = line[s + 1:]

    return labels_dict


def pad_images(images, im_length, im_height):
    padded_images = []
    for image in images:
        padded_image = np.zeros([im_length, im_height])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                padded_image[i, j] = image[i, j]

        padded_images.append(padded_image)

    return padded_images


class Data:

    def __init__(self, images_dir_path, labels_txt_path):

        self.encode = 1
        self.image_dir_path = images_dir_path
        self.images_path = np.array(os.listdir(images_dir_path))
        self.labels_dict = get_labels_dict(labels_txt_path)

        self.im_length = 384
        self.im_height = 28

    def generator(self, batch_size):
        instance_id = range(len(self.images_path))
        while True:
            try:
                batch_ids = random.sample(instance_id, batch_size)  # list of random ids

                variable_size_images_batch = [mpimg.imread(  # images to be padded
                    self.image_dir_path + self.images_path[id_])
                    for id_ in batch_ids]

                images_batch = pad_images(variable_size_images_batch, self.im_length, self.im_height)

                labels_batch = [self.labels_dict[image_path]
                                for image_path in self.images_path[batch_ids]]

                yield (images_batch, labels_batch)
            except Exception as e:
                print('EXCEPTION OMG')
                print(e)
                yield None, None


if __name__ == "__main__":
    root = "/Users/charles/Data/Hamelin/"
    images_test_dir = root + "TST/test/"
    labels_test_txt = root + "test.txt"

    data = Data(images_test_dir, labels_test_txt)

    gen = data.generator(3)
    a = gen.__next__()
    images = a[0]

    plt.imshow(images[0])
    plt.show()

    print('fin')
