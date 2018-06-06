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
