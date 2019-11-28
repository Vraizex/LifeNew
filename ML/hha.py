
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
from tqdm import tqdm_notebook, tqdm
from skimage import ellipse, polygon
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout
from keras.losses import binary_crossentropy
import tensorflow as tf
import keras as keras

from keras import backend as K

from tqdm import tqdm_notebook
w_size = 256
train_num = 8192
train_x = np.zeros((train_num, w_size, w_size,3), dtype='float32')
train_y = np.zeros((train_num, w_size, w_size,1), dtype='float32')

img_l = np.random.sample((w_size, w_size, 3))*0.5
img_h = np.random.sample((w_size, w_size, 3))*0.5 + 0.5

radius_min = 10
radius_max = 30


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def get_iou_vector(A, B):
    # Numpy version

    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)

        # deal with empty mask first
        if true == 0:
            metric += (pred == 0)
            continue

        # non empty mask case.  Union is never empty
        # hence it is safe to divide by its number of pixels
        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union

        # iou metrric is a stepwise approximation of the real iou over 0.5
        iou = np.floor(max(0, (iou - 0.45) * 20)) / 10

        metric += iou

    # teake the average over all images in batch
    metric /= batch_size
    return metric


def my_iou_metric(label, pred):
    # Tensorflow version
    return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)


from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({'bce_dice_loss': bce_dice_loss})
get_custom_objects().update({'dice_loss': dice_loss})
get_custom_objects().update({'dice_coef': dice_coef})
get_custom_objects().update({'my_iou_metric': my_iou_metric})


def next_pair():
    p = np.random.sample() - 0.5  # пока не успользуем
    # r,c - координаты центра эллипса
    r = np.random.sample() * (w_size - 2 * radius_max) + radius_max
    c = np.random.sample() * (w_size - 2 * radius_max) + radius_max
    # большой и малый радиусы эллипса
    r_radius = np.random.sample() * (radius_max - radius_min) + radius_min
    c_radius = np.random.sample() * (radius_max - radius_min) + radius_min
    rot = np.random.sample() * 360  # наклон эллипса
    rr, cc = ellipse(
        r, c,
        r_radius, c_radius,
        rotation=np.deg2rad(rot),
        shape=img_l.shape
    )  # получаем все точки эллипса

    # красим пиксели моря/фона в шум от 0.5 до 1.0
    img = img_h.copy()
    # красим пиксели эллипса в шум от 0.0  до 0.5
    img[rr, cc] = img_l[rr, cc]

    msk = np.zeros((w_size, w_size, 1), dtype='float32')
    msk[rr, cc] = 1.  # красим пиксели маски эллипса

    return img, msk


for k in range(train_num):  # генерация всех img train
    img, msk = next_pair()

    train_x[k] = img
    train_y[k] = msk

fig, axes = plt.subplots(2, 10, figsize=(20, 5))  # смотрим на первые 10 с масками
for k in range(10):
    axes[0, k].set_axis_off()
    axes[0, k].imshow(train_x[k])
    axes[1, k].set_axis_off()
    axes[1, k].imshow(train_y[k].squeeze())



input_layer = Input((w_size, w_size, 3))
output_layer = build_model(input_layer, 16)
model = Model(input_layer, output_layer)
model.compile(loss=bce_dice_loss, optimizer=Adam(lr=1e-3), metrics=[my_iou_metric])
model.save_weights('./keras.weights')

while True:
    history = model.fit(train_x, train_y,
                        batch_size=32,
                        epochs=1,
                        verbose=1,
                        validation_split=0.1
                       )
    if history.history['my_iou_metric'][0] > 0.75:
        break