import numpy as np
import random
import os
import matplotlib.pyplot as plt


class RData:

    def rinit(self, im_length=64, im_height=64, nb_data=32):        #random init

        self.im_length = im_length
        self.im_height = im_height
        self.nb_data = nb_data
        self.truths = np.array([[[random.randint(0, 1)
                                  for _ in range(30)]
                                 for _ in range(self.im_length)]
                                for _ in range(self.nb_data)])

        self.images = np.array([10 * np.random.rand(self.im_length, self.im_height) for _ in range(self.nb_data)])


class Data:

    def __init__(self, root, images_dir, labels_txt):

        self.encode = 1

        self.images_lst = os.listdir(root + images_dir)

        labels_dict = self.get_labels_dict(root + labels_txt)

    def get_labels_dict(self, labels_txt):
        txt = open(labels_txt).read()
        s_index = [i for i,x in enumerate(txt) if x==" "]           #indices des espaces dans le texte
        n_index = [i for i,x in enumerate(txt) if x=="\n"]          #indice des retours a la ligne das le text

        i = 0
        lines = []
        d = 0
        f = n_index[0]
        line = txt[d:f]
        for i in range(len(n_index)):
            i += 1
            d = n_index[i-1] + 1
            f = n_index[i]
            print(i)
            lines.append(line)
            line = txt[d:f]



        print("fin")

    def generator(self, batch_size):
        instance_id = range(self.nb_data)
        while True:
            try:
                batch_ids = random.sample(instance_id, batch_size)
                yield (np.array(self.images[batch_ids], dtype=int).reshape(batch_size, self.im_length, self.im_height, 1),
                       np.array(self.truths[batch_ids]))
            except Exception as e:
                print('EXCEPTION OMG')
                print(e)
                yield None, None


if __name__ == "__main__":

    root = "/home/abrecadabrac/Modèles/data/Hamelin_full/"
    images_test_dir = "TST/test"
    labels_test_txt = "test.txt"


    data = Data(root, images_test_dir, labels_test_txt)

    print('fin')