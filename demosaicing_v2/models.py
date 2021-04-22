import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import Config

class ModelBuilder():
    def __init__(self):
        cf = Config()
        self.input_shape = (cf.img_size, cf.img_size, cf.n_channels)
        self.n_channels = cf.n_channels
        # print(self.input_shape)


    def conv_block(self, x, filters, kernel_size, stride, padding):
        x = layers.Conv2D(filters, kernel_size,strides=stride, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, kernel_size,strides=stride, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x




    def build_model_v1(self):
        inputs = layers.Input(shape=self.input_shape)
        x = self.conv_block(inputs, 32, 7, 1, 'same')
        x = self.conv_block(x, 64, 5, 1, 'same')
        x = self.conv_block(x, 64, 3, 1, 'same')
        x = self.conv_block(x, 32, 3, 1, 'same')
        con = layers.Concatenate()([inputs,x])
        out = layers.Conv2D(3, 1, 1, 'same')(con)
        model = keras.Model([inputs], [out])
        # model.summary()
        return model

    def build_model_v1_s2(self):
        inputs = layers.Input(shape=self.input_shape)
        x = self.conv_block(inputs, 32, 9, 1, 'same')
        x = self.conv_block(x, 64, 9, 1, 'same')
        x = layers.MaxPooling2D()(x)
        x = self.conv_block(x, 64, 3, 1, 'same')
        x = layers.MaxPooling2D()(x)
        x = self.conv_block(x, 32, 3, 1, 'same')
        x = layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x)
        x = self.conv_block(x, 32, 3, 1, 'same')
        x = layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x)
        x = self.conv_block(x, 32, 3, 1, 'same')
        con = layers.Concatenate()([inputs,x])
        out = layers.Conv2D(3, 1, 1, 'same')(con)
        model = keras.Model([inputs], [out])
        # model.summary()
        return model