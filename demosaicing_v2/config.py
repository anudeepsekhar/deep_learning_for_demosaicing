class Config():
    def __init__(self):
        self.img_size = 128
        self.n_channels = 3
        self.input_shape = [None,None,3]
        self.data_dir = '/home/aicenter/Projects/demosaicing/deep_learning_for_demosaicing/data/CUB_200_2011/images'
        self.batch_size = 16
        self.plot_freq = 2
