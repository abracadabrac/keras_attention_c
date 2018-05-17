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


def pad_images(images, im_height, im_length):
    padded_images = []
    for image in images:
        padded_image = np.zeros([im_height, im_length])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                padded_image[i, j] = image[i, j]

        padded_images.append(np.rot90(padded_image, k=-1))

    return padded_images


class Data:

    def __init__(self, images_dir_path, labels_txt_path):

        self.im_height = 28
        self.im_length = 384
        self.lb_length = 0  # normalized length of the label, its value is initialized when calling self.get_encoding_dict
        self.encode = 1
        self.image_dir_path = images_dir_path
        self.labels_txt_path = labels_txt_path

        self.images_path = np.array(os.listdir(images_dir_path))
        self.labels_dict = self.get_labels_dict()  # {image path: text sequence}
        self.encoding_dict, self.decoding_dict = self.get_encoding_dicts()  # {char: one hot encoded char}, { index: char}

        self.vocab_size = len(self.encoding_dict)  # total numbers of different characters within the vocabulary, (28)

    def generator(self, batch_size):
        instance_id = range(len(self.images_path))
        while True:
            try:
                batch_ids = random.sample(instance_id, batch_size)  # list of random ids

                variable_size_images_batch_list = [mpimg.imread(  # images to be padded
                    self.image_dir_path + self.images_path[id_])
                    for id_ in batch_ids]
                images_batch_list = pad_images(variable_size_images_batch_list, self.im_height, self.im_length)

                images_batch = np.array(images_batch_list).reshape(batch_size, self.im_length, self.im_height, 1)

                variable_size_words_batch_list = [self.labels_dict[image_path]              # list of variable-size non-encoded words
                                                  for image_path in self.images_path[batch_ids]]

                labels_batch = self.encode_label(variable_size_words_batch_list)            # encode and normalize size

                yield (images_batch, labels_batch)
            except Exception as e:
                print('EXCEPTION OMG')
                print(e)
                yield None, None

    def get_labels_dict(self):
        labels_dict = {}
        txt = open(self.labels_txt_path).read()  # text contenant les labels et les noms des fichiers images
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

    def get_encoding_dicts(self):
        chars_list = []
        for word in self.labels_dict.values():
            if len(word) > self.lb_length:
                self.lb_length = len(word)
            for char in word:
                if char not in chars_list:
                    chars_list.append(char)

        encoding_dict = {}
        for index, char in enumerate(chars_list):
            char_encoded = list(np.zeros(len(chars_list)))
            char_encoded[index] = 1.
            encoding_dict[char] = char_encoded

        decoding_dict = {list(char_encoded).index(1): char
                         for (char, char_encoded) in encoding_dict.items()}

        return encoding_dict, decoding_dict

    def encode_label(self, labels):
        encoded_labels = []
        for label in labels:
            encoded_label = []
            for char in label:
                encoded_label.append(self.encoding_dict[char])
            encoded_labels.append(encoded_label)

        encoded_labels = np.array([xi + [list(np.zeros(self.vocab_size))] * (self.lb_length - len(xi)) for xi in encoded_labels])

        return encoded_labels


def test_generator():
    root = "/Users/charles/Data/Hamelin/"
    images_test_dir = root + "TST/test/"
    labels_test_txt = root + "test.txt"

    data = Data(images_test_dir, labels_test_txt)

    gen = data.generator(3)
    images, labels = gen.__next__()

    plt.imshow(images[0, :, :, 0])
    plt.show()


if __name__ == "__main__":
    test_generator()

    print('fin')
