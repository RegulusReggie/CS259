import os
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K

img_rows = 96
img_cols = 96

def dice(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def jaccard(y_true, y_pred, smooth=1.):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true) + K.sum(y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def lossfunc(y_true, y_pred):
    return - 1 * jaccard(y_true, y_pred)
    # return - 0.7 * jaccard(y_true, y_pred) - 0.3 * dice(y_true, y_pred)

def shortcut(inputs, residual):

    stride_width = inputs._keras_shape[1] / residual._keras_shape[1]
    stride_height = inputs._keras_shape[2] / residual._keras_shape[2]
    equal_ch = inputs._keras_shape[3] == residual._keras_shape[3]
    if stride_width > 1 or stride_height > 1 or not equal_ch:
        inputs = Conv2D(residual._keras_shape[3], 1, strides=(stride_width, stride_height), padding='valid')(inputs)

    return Add()([inputs, residual])

def rblock_concat(inputs, n_filters, kernel_size):
    conv = Conv2D(n_filters, kernel_size, padding='same')(inputs)
    conv = BatchNormalization(axis=3)(conv)
    residual = shortcut(inputs, conv)
    return ReLU()(residual)

def rblock_upconv(inputs, concat, n_filters, kernel_size):
    up = Conv2D(n_filters, 2, activation='relu', padding='same')(UpSampling2D()(inputs))
    up = BatchNormalization(axis=3)(up)
    concat = Concatenate(axis=3)([rblock_concat(concat, n_filters, kernel_size), up])
    # concat = Concatenate(axis=3)([concat, up])
    conv = Conv2D(n_filters, kernel_size, activation='relu', padding='same')(concat)
    conv = BatchNormalization(axis=3)(conv)
    conv = Conv2D(n_filters, kernel_size, activation='relu', padding='same')(conv)
    # conv = Conv2D(n_filters, kernel_size, padding='same')(conv)
    conv = BatchNormalization(axis=3)(conv)

    # conv = shortcut(concat, conv)
    # return ReLU()(conv)
    return conv

def unet():
    inputs = Input((img_rows, img_cols, 3))
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization(axis=3)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization(axis=3)(conv2)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization(axis=3)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization(axis=3)(conv3)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization(axis=3)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) 
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization(axis=3)(conv4)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization(axis=3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(Dropout(0.2)(conv4))
    
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization(axis=3)(conv5)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization(axis=3)(conv5)
    
    middle = Dropout(0.2)(conv5)
    
    # up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D()(middle)) 
    # up6 = BatchNormalization(axis=3)(up6)
    # concat6 = Concatenate(axis=3)([conv4, up6])
    # conv6 = Conv2D(512, 3, activation='relu', padding='same')(concat6)
    # conv6 = BatchNormalization(axis=3)(conv6)
    # conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
    # conv6 = BatchNormalization(axis=3)(conv6)

    conv6 = rblock_upconv(middle, conv4, 512, 3)
    
    # up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D()(conv6)) 
    # up7 = BatchNormalization(axis=3)(up7)
    # concat7 = Concatenate(axis=3)([conv3, up7])
    # conv7 = Conv2D(256, 3, activation='relu', padding='same')(concat7)
    # conv7 = BatchNormalization(axis=3)(conv7)
    # conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
    # conv7 = BatchNormalization(axis=3)(conv7)

    conv7 = rblock_upconv(conv6, conv3, 256, 3)
    
    # up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D()(conv7)) 
    # up8 = BatchNormalization(axis=3)(up8)
    # concat8 = Concatenate(axis=3)([conv2, up8])
    # conv8 = Conv2D(128, 3, activation='relu', padding='same')(concat8)
    # conv8 = BatchNormalization(axis=3)(conv8)
    # conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    # conv8 = BatchNormalization(axis=3)(conv8)

    conv8 = rblock_upconv(conv7, conv2, 128, 3)
    
    # up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D()(conv8)) 
    # up9 = BatchNormalization(axis=3)(up9)
    # concat9 = Concatenate(axis=3)([conv1, up9])
    # conv9 = Conv2D(64, 3, activation='relu', padding='same')(concat9)
    # conv9 = BatchNormalization(axis=3)(conv9)
    # conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    # conv9 = BatchNormalization(axis=3)(conv9)

    conv9 = rblock_upconv(conv8, conv1, 64, 3)
    
    conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9)
    
    model = Model(inputs=[inputs], outputs=[conv10])
    
    model.compile(optimizer=RMSprop(lr=1e-4), loss=lossfunc, metrics=[jaccard])
    
    
    return model

get_unet = unet