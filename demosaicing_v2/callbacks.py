import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np


class VizCallback(keras.callbacks.Callback):
    def __init__(self, get_data, plot_data, freq):
        super(VizCallback, self).__init__()
        self.freq = freq
        self.test_cfa, self.test_img = get_data()
        self.plot_data = plot_data

    def on_epoch_end(self, epoch, logs=None):
        if epoch%self.freq==0:
            pred = self.model.predict(self.test_cfa)
            self.plot_data(self.test_img, self.test_cfa, epoch, self.model, )
