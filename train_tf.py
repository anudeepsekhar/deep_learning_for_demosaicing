#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import tensorflow.keras.backend as K

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import shuffle

from glob import glob
from tqdm import tqdm

import albumentations as A
import argparse
from config import Config 
from models import ModelBuilder
from callbacks import VizCallback

from sklearn.model_selection import train_test_split
import argparse

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

cf = Config()
#%%
def calc_psnr(y_true,y_pred):
    max_pixel = 1.0
    return np.mean((10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303)

def remosaic(img):
    Ny, Nx, Nc = img.shape
    B = np.zeros([2*Ny, 2*Nx, 3])
    for i in range(1,Ny):
        for j in range(1,Nx):
            B[2*i-1,2*j-1,0] = img[i,j,0]
            B[2*i,2*j,2] = img[i,j,2]
            B[2*i,2*j-1,1] = img[i,j,1]
            B[2*i-1,2*j,1] = img[i,j,1]
    return B

def parse_data(path, train):
    path = path.decode()
    img = cv2.imread(path)
    img = cv2.resize(img, (cf.img_size, cf.img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_down = cv2.resize(img, (cf.img_size//2, cf.img_size//2))
    img_down = cv2.blur(img_down, (1,1))
    cfa = remosaic(img_down)
    assert cfa.shape==img.shape

    cfa = cfa.astype(np.float32)/255.0
    img = img.astype(np.float32)/255.0

    return cfa, img

def load_train_data(path):
    cfa, rgb = tf.numpy_function(parse_data, [path, True], [tf.float32, tf.float32])
    cfa.set_shape(cf.input_shape)
    rgb.set_shape(cf.input_shape)
    return cfa, rgb

def load_val_data(path):
    cfa, rgb = tf.numpy_function(parse_data, [path, False], [tf.float32, tf.float32])
    cfa.set_shape(cf.input_shape)
    rgb.set_shape(cf.input_shape)
    return cfa, rgb

def get_test_data():
    for batch in train_ds.take(1):
        images, target = batch
    return images, target


def plot_results(org_image, cfa, model = None):
    if model is None:
        fig, ax = plt.subplots(1,2, figsize=(4,5))
        ax[0].imshow(org_image.numpy()[0])
        ax[1].imshow(cfa.numpy()[0])
        for a in ax: a.axis('off') 
        plt.show()
    else:
        output = model.predict(cfa)
        psnr = calc_psnr(org_image, output)
        fig, ax = plt.subplots(1,3, figsize=(12,8))
        plt.title(f'PSNR: {psnr}')
        ax[0].imshow(org_image.numpy()[0])
        ax[1].imshow(cfa.numpy()[0])
        ax[2].imshow(output[0])
        for a in ax: a.axis('off') 
        plt.show()



#%%
parser = argparse.ArgumentParser(description='Demosaicnet Trainer')
parser.add_argument('stage', help='stage of neural network you want to train')
parser.add_argument('resume', help='resume training from latest checkpoint')

args = parser.parse_args()

#%%

image_paths = sorted(glob(os.path.join(cf.data_dir, '*/*.jpg')))
print(f'Number of images in data dir: {len(image_paths)}')
train_paths, test_paths = train_test_split(image_paths, test_size=0.2, shuffle=True)
print(f'Number of images in train: {len(train_paths)}')
print(f'Number of images in test: {len(test_paths)}')

#%%

train_ds = tf.data.Dataset.from_tensor_slices(train_paths)
train_ds = train_ds.map(load_train_data, num_parallel_calls=tf.data.AUTOTUNE)
# train_ds = train_ds.shuffle(buffer_size=len(train_ds))
train_ds = train_ds.batch(batch_size=cf.batch_size)

val_ds = tf.data.Dataset.from_tensor_slices(test_paths)
val_ds = val_ds.map(load_val_data)
val_ds = val_ds.batch(batch_size=cf.batch_size)

# %%
i = 2
for batch in train_ds.take(1):
    images, target = batch
    print(images.shape)
fig, ax = plt.subplots(1,2)
ax[0].imshow(images.numpy()[i][:32,:32,:])
ax[1].imshow(target.numpy()[i][:32,:32,:])


# %%
mb = ModelBuilder()
model = mb.build_model_v1()
plot_results(target, images, model)

# %%
epochs = 10
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.00001)

viz_cb = VizCallback(get_test_data, plot_results, cf.plot_freq)


checkpoint_path = "checkpoints_MS_v2_s1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=False,
    save_weights_only=True, mode='auto', save_freq='epoch',
    options=None
)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(
    optimizer=optimizer, loss=keras.losses.mean_absolute_error, metrics=[keras.metrics.RootMeanSquaredError()]    
) 

# %%
hist = model.fit(train_ds, epochs=epochs, validation_data=val_ds, batch_size=cf.batch_size, verbose=1, callbacks=[ckpt_callback, reduce_lr])

# %%
#train second stage
model1 = model
model1.trainable = False

model2 = mb.build_model_v1_s2()
# model2.summary()
out = model2(model1.output)
model = keras.Model([model1.input], [out])

model.summary()
# %%
epochs = 10
checkpoint_path = "checkpoints_MS_v2_s2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=False,
    save_weights_only=True, mode='auto', save_freq='epoch',
    options=None
    )

model.compile(
    optimizer=optimizer, loss=keras.losses.mean_squared_error, metrics=[keras.metrics.RootMeanSquaredError()]    
)

# %%
hist = model.fit(train_ds, epochs=epochs, validation_data=val_ds, batch_size=cf.batch_size, verbose=1, callbacks=[ckpt_callback, reduce_lr])
# %%
