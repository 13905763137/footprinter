import tensorflow as tf
import os
import numpy as np 
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D, Dropout, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, LearningRateScheduler
from tensorflow.keras import Sequential
from keras import backend as K
from datagen import generator
import argparse

def unit_conv_layer(nfilters, size, activation = "relu", padding = "same", kernel_initializer = "he_normal"):
    '''
    Unit Sequential Convolution Layer Used in UNET
    args:
         size: size of the convolution filter
         nfilters : number of channels
         default activation = relu
         default padding = same
         default kernel_initializer = he_normal (look at https://www.deeplearning.ai/ai-notes/initialization/)
    '''
    block = Sequential()
    block.add(Conv2D(filters = nfilters, 
                     kernel_size = size, 
                     activation = activation, 
                     padding = padding, 
                     kernel_initializer = kernel_initializer)) 
    
    return block

def unit_upsampler(size = (2,2)):
    '''
    Unit upsampler in the decoder part of UNET
    args:
         size: size of upsampling (size[0], size[1]);
         default (2, 2)
    '''
    block = Sequential()
    block.add(UpSampling2D(size))
    
    return block

def baseline_unet_model(input_shape = (128, 128, 3)):
   
    inputs = Input(input_shape)
    conv1 = unit_conv_layer(64, 3)(inputs)
    conv1 = unit_conv_layer(64, 3)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = unit_conv_layer(128, 3)(pool1)
    conv2 = unit_conv_layer(128, 3)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = unit_conv_layer(256, 3)(pool2)
    conv3 = unit_conv_layer(256, 3)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = unit_conv_layer(512, 3)(pool3)
    conv4 = unit_conv_layer(512, 3)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = unit_conv_layer(1024, 3)(pool4)
    conv5 = unit_conv_layer(1024, 3)(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = unit_upsampler()(drop5)
    up6 = unit_conv_layer(512, 2)(up6)
    merge6 = concatenate([drop4, up6], axis = 3)
    conv6 = unit_conv_layer(512, 3)(merge6)
    conv6 = unit_conv_layer(512, 3)(conv6)

    up7 = unit_upsampler()(conv6)
    up7 = unit_conv_layer(256, 2)(up7)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = unit_conv_layer(256, 3)(merge7)
    conv7 = unit_conv_layer(256, 3)(conv7)

    up8 = unit_upsampler()(conv7)
    up8 = unit_conv_layer(128, 2)(up8)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = unit_conv_layer(128, 3)(merge8)
    conv8 = unit_conv_layer(128, 3)(conv8)

    up9 = unit_upsampler()(conv8)
    up9 = unit_conv_layer(64, 2)(up9)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = unit_conv_layer(64, 3)(merge9)
    conv9 = unit_conv_layer(64, 3)(conv9)
    conv9 = unit_conv_layer(2, 3)(conv9)
    preds = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = tf.keras.Model(inputs = inputs, outputs = preds)

    model.summary()

    return model
