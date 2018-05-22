import numpy as np
import random
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def pad_images(images, im_height, im_length):
    padded_images = []
    for image in images:
        padded_image = np.zeros([im_height, im_length])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                padded_image[i, j] = image[i, j]

        padded_images.append(np.rot90(padded_image, k=-1))

    return padded_images


def get_labels_dict(labels_txt_path):
    """
    :param labels_txt_path:
    :return: a dictionary of corresponce between image files and associated labels

    ex :    { 'NEAT_0-19word12561420170630155102_005_w005.png': 'NICOSIA',
              'NEAT_0-19word12583420170703115206_001_w007.png': 'IDEAS',
              'NEAT_0-19word12534620170629153942_005_w008.png': 'WRITINGS', ... }
    """
    labels_dict = {}
    txt = open(labels_txt_path).read()  # text contenant les labels et les noms des fichiers images
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


def pred2OneHot(probability_vector):
    """
    Neural nets predict vectors of probability for each element of the sequence over the vocabulary.
    this function permits to get the most likely label according to the net's prediction

    :param probability_vector:
    :return: most likely encoded chars

    ex : [ [.2, .3, .5], [.1, .7, .2] ] -> [ [0, 0, 1], [0, 1, 0] ]
    """
    print('fin')


class Data:

    def __init__(self, images_dir_path, labels_txt_path):

        self.im_height = 28
        self.im_length = 384
        self.lb_length = 16  # normalized length of the encoded elements of the labels
        self.image_dir_path = images_dir_path
        self.labels_txt_path = labels_txt_path

        self.labels_dict = get_labels_dict(labels_txt_path)
        self.encoding_dict = np.load('/Users/charles/Workspace/text_attention/data/encoding_dict_Hamelin.npy').item()
        self.decoding_dict = np.load('/Users/charles/Workspace/text_attention/data/decoding_dict_Hamelin.npy').item()

        self.images_path = np.array(os.listdir(images_dir_path))

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

                variable_size_words_batch_list = [self.labels_dict[image_path]
                                                  # list of variable-size non-encoded words
                                                  for image_path in self.images_path[batch_ids]]

                labels_batch = self.encode_label(variable_size_words_batch_list)  # encode and normalize size

                yield (images_batch, labels_batch)
            except Exception as e:
                print('EXCEPTION OMG')
                print(e)
                yield None, None

    def encode_label(self, labels):
        """
        :param labels: list of strings of variable length.
        :return: np.array representing a sequence of encoded elements.
        the sequences are completed with zero vectors.

        encoded_labels.shape = (batch_size, self.lb_length, self.vocab_size)
        """
        encoded_labels = []
        for label in labels:
            encoded_label = []
            for char in label:
                encoded_label.append(self.encoding_dict[char])
            encoded_labels.append(encoded_label)

        encoded_labels = np.array(
            [xi + [list(np.zeros(self.vocab_size))] * (self.lb_length - len(xi)) for xi in encoded_labels])

        return encoded_labels

    def decode_labels(self, labels):
        """
        :param labels: np.array of list representing a sequence of encoded labels.
        :return: the corresponding list of strings padded with "_" for null elements
        """
        decoded_labels = []
        for label in labels:
            decoded_label = ''
            for e in label:
                if np.sum(e) == 1:
                    decoded_label += self.decoding_dict[list(e).index(1.)]
                else:
                    decoded_label += "_"
            decoded_labels.append(decoded_label)
        return decoded_labels


def main1():
    root = "/Users/charles/Data/Hamelin/"
    images_test_dir = root + "TST/test/"
    labels_test_txt = root + "test.txt"

    data = Data(images_test_dir, labels_test_txt)

    gen = data.generator(12)
    images, labels = gen.__next__()

    decoded_labels = data.decode_labels(labels)

    print(decoded_labels)


if __name__ == "__main__":
    main1()

    print('fin')
