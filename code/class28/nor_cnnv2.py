import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.python.keras.backend import argmax
from tensorflow import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential, load_model, Model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Input, concatenate, Maximum, Lambda
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.applications.xception import Xception
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import np_utils

import os
# import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#    x = Lambda(lambda x : x[-1])(x)

def f_layer(Input_tensor):
    x = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(Input_tensor)
    x = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x)
    x = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x)
    return x

def CNN_28(input_size=(299,299,3)):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_size)
    x = base_model.output
    f = GlobalAveragePooling2D(data_format='channels_last')(x)
    f = Dropout(0.5)(f)  
    #1
    x1 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x1 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x1)
    x1 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x1)
    #2
    x2 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x2 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x2)
    x2 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x2)
    #3
    x3 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x3 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x3)
    x3 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x3)
    #4
    x4 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x4 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x4)
    x4 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x4)
    #5
    x5 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x5 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x5)
    x5 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x5)
    #6
    x6 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x6 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x6)
    x6 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x6)
    #7
    x7 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x7 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x7)
    x7 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x7)
    #8
    x8 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x8 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x8)
    x8 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x8)
    #9
    x9 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x9 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x9)
    x9 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x9)
    #10
    x10 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x10 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x10)
    x10 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x10)
    #11
    x11 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x11 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x11)
    x11 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x11)
    #12
    x12 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x12 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x12)
    x12 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x12)
    #13
    x13 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x13 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x13)
    x13 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x13)
    #14
    x14 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x14 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x14)
    x14 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x14)
    #15
    x15 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x15 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x15)
    x15 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x15)
    #16
    x16 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x16 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x16)
    x16 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x16)
    #17
    x17 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x17 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x17)
    x17 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x17)
    #18
    x18 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x18 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x18)
    x18 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x18)
    #19
    x19 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x19 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x19)
    x19 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x19)
    #20
    x20 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x20 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x20)
    x20 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x20)
    #21
    x21 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x21 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x21)
    x21 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x21)
    #22
    x22 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x22 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x22)
    x22 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x22)
    #23
    x23 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x23 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x23)
    x23 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x23)
    #24
    x24 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x24 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x24)
    x24 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x24)
    #25
    x25 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x25 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x25)
    x25 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x25)
    #26
    x26 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x26 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x26)
    x26 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x26)
    #27
    x27 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x27 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x27)
    x27 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x27)
    #28
    x28 = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(f)
    x28 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(x28)
    x28 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(x28)
    co = concatenate([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28])
    
    model = Model(inputs = base_model.input, outputs = co)
    return model

model = CNN_28()  
model.summary()
