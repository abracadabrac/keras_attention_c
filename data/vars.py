

"""
In this class you save the paths relative to your own data structure
"""

class Vars:

    def __init__(self):

        self.root = "/Users/charles/Data/Hamelin/"

        self.images_train_dir = self.root + "TRAIN/train/"
        self.images_valid_dir = self.root + "VAL/valid/"
        self.images_test_dir = self.root + "TST/test/"

        self.labels_train_txt = self.root + "train.txt"
        self.labels_valid_txt = self.root + "valid.txt"
        self.labels_test_txt = self.root + "test.txt"

        self.encoding_dict = '/Users/charles/Workspace/text_attention/data/encoding_dict_Hamelin.npy'
        self.decoding_dict = '/Users/charles/Workspace/text_attention/data/decoding_dict_Hamelin.npy'
