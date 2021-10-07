import numpy as np
from tensorflow.python.keras.layers import Conv2D,MaxPooling2D,Activation,AveragePooling2D,BatchNormalization,add,Input,ZeroPadding2D,Flatten,Dense,Dropout\
, Reshape, Lambda, concatenate, GlobalAveragePooling2D
from tensorflow.python.keras.activations import softmax,sigmoid
from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2

def res_block(Input_tensor, kernel_size, filters, stage, block, use_bias=True):
        nb_filter1,nb_filter2 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        x = Conv2D(nb_filter1, (kernel_size, kernel_size), padding = 'same', name = conv_name_base + '2a', use_bias=use_bias, kernel_initializer = 'he_normal')(Input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding = 'same', name = conv_name_base + '2b', use_bias=use_bias, kernel_initializer = 'he_normal')(x)
        x = BatchNormalization()(x)
        x = add([x, Input_tensor])
        x = Activation('relu')(x)
        return x
def conv_block(Input_tensor, kernel_size, filters, stage, block, strides=(2,2), use_bias=True):
        nb_filter1,nb_filter2 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        
        x = Conv2D(nb_filter1, (kernel_size, kernel_size), padding = 'same', strides = strides, name = conv_name_base + '2a', use_bias=use_bias, kernel_initializer = 'he_normal')(Input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
        x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding = 'same', name = conv_name_base + '2b', use_bias=use_bias, kernel_initializer = 'he_normal')(x)
        x = BatchNormalization()(x)
                
        shortcut = Conv2D(nb_filter2, (1,1), strides = strides, name = conv_name_base + '1', use_bias=use_bias, kernel_initializer = 'he_normal')(Input_tensor)
        shortcut = BatchNormalization()(shortcut)
        
        x = add([x, shortcut])
        x = Activation('relu')(x)
        return x
     
def ResNet(input_size=(512,512,3)):
        input_ly = Input(shape = input_size)
        x = ZeroPadding2D((3,3))(input_ly)
        x = Conv2D(64, kernel_size=(7,7), strides = 2, padding = 'valid', kernel_initializer = 'he_normal')(x)
        pool1 = MaxPooling2D((3,3), strides=2, padding = 'same')(x)
        
        #stage2
        x = conv_block(pool1, 3, [64, 64], stage=2, block='a', strides=(1,1))
        x = res_block(x, 3, [64, 64], stage=2, block='b')
        #stage3
        x = conv_block(x, 3, [128, 128], stage=3, block='a', strides=(2,2))
        x = res_block(x, 3, [128, 128], stage=3, block='b')
        #stage4
        x = conv_block(x, 3, [256, 256], stage=4, block='a', strides=(2,2))
        x = res_block(x, 3, [256, 256], stage=4, block='b')
        #stage5
        x = conv_block(x, 3, [512, 512], stage=5, block='a', strides=(2,2))
        x = res_block(x, 3, [512, 512], stage=5, block='b')
        
        x_end = AveragePooling2D((7,7), padding = 'valid')(x)
        x = Flatten()(x_end)
        # x = Dropout(0.4)(x)
        x = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(x)
        # x = Dropout(0.4)(x)
        x = Dense(256, activation = 'relu', kernel_initializer = 'he_normal')(x)
        # x = Dropout(0.4)(x)
        x = Dense(28, kernel_initializer = 'he_normal')(x)
        
        x_sec = Flatten()(x_end)
        # x_sec= Dropout(0.4)(x_sec)
        x_sec= Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(x_sec)
        # x_sec= Dropout(0.4)(x_sec)
        x_sec= Dense(256, activation = 'relu', kernel_initializer = 'he_normal')(x_sec)
        # x_sec= Dropout(0.4)(x_sec)
        x_sec= Dense(28, kernel_initializer = 'he_normal')(x_sec)
        x_sec = Reshape((28,1))(x_sec)
        x          = Reshape((28,1))(x)
        x_lest = concatenate([x,x_sec],axis=-1)
        x_lest = Lambda(lambda x : softmax(x,axis = -1))(x_lest)
        
        model = Model(inputs = input_ly, outputs = x_lest)
        return model
    
def ResNet_sigmoid(input_size=(512,512,3)):
        input_ly = Input(shape = input_size)
        x = ZeroPadding2D((3,3))(input_ly)
        x = Conv2D(64, kernel_size=(7,7), strides = 2, padding = 'valid', kernel_initializer = 'he_normal')(x)
        pool1 = MaxPooling2D((3,3), strides=2, padding = 'same')(x)
        
        #stage2
        x = conv_block(pool1, 3, [64, 64], stage=2, block='a', strides=(1,1))
        x = res_block(x, 3, [64, 64], stage=2, block='b')
        #stage3
        x = conv_block(x, 3, [128, 128], stage=3, block='a', strides=(2,2))
        x = res_block(x, 3, [128, 128], stage=3, block='b')
        #stage4
        x = conv_block(x, 3, [256, 256], stage=4, block='a', strides=(2,2))
        x = res_block(x, 3, [256, 256], stage=4, block='b')
        #stage5
        x = conv_block(x, 3, [512, 512], stage=5, block='a', strides=(2,2))
        x = res_block(x, 3, [512, 512], stage=5, block='b')
        
        x_end = AveragePooling2D((7,7), padding = 'valid')(x)
        x = Flatten()(x_end)
        x = Dropout(0.4)(x)
        x = Dense(1024, activation = 'relu', kernel_initializer = 'he_normal')(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation = 'relu', kernel_initializer = 'he_normal')(x)
        x = Dropout(0.4)(x)
        x = Dense(28, activation = 'sigmoid',kernel_initializer = 'he_normal')(x)
        
        model = Model(inputs = input_ly, outputs = x)
        return model

def my_Xception (input_shape = (299,299,3)):
    base_model = Xception(weights='imagenet', include_top=False,
                         input_shape=input_shape)
    x_fin = base_model.output
    x  = GlobalAveragePooling2D(data_format='channels_last')(x_fin)
    # x = Dropout(0.4)(x)
    x = Dense(256, activation = 'relu', kernel_initializer = 'he_normal')(x)
    # x = Dropout(0.4)(x)
    x = Dense(28, kernel_initializer = 'he_normal')(x)
    
    x_sec = GlobalAveragePooling2D(data_format='channels_last')(x_fin)
    # x_sec= Dropout(0.4)(x_sec)
    x_sec= Dense(256, activation = 'relu', kernel_initializer = 'he_normal')(x_sec)
    # x_sec= Dropout(0.4)(x_sec)
    x_sec= Dense(28, kernel_initializer = 'he_normal')(x_sec)
    x_sec = Reshape((28,1))(x_sec)
    x          = Reshape((28,1))(x)
    x_lest = concatenate([x,x_sec],axis=-1)
    x_lest = Lambda(lambda x : sigmoid(x))(x_lest)
    
    model = Model(inputs = base_model.input, outputs = x_lest)
    return model

def my_InceptionResNet (input_shape = (299,299,3)):
    base_model = InceptionResNetV2(weights='imagenet', include_top=False,
                         input_shape=input_shape)
    x_fin = base_model.output
    x  = GlobalAveragePooling2D(data_format='channels_last')(x_fin)
    # x = Dropout(0.4)(x)
    x = Dense(256, activation = 'relu', kernel_initializer = 'he_normal')(x)
    # x = Dropout(0.4)(x)
    x = Dense(28, kernel_initializer = 'he_normal')(x)
    
    x_sec = GlobalAveragePooling2D(data_format='channels_last')(x_fin)
    # x_sec= Dropout(0.4)(x_sec)
    x_sec= Dense(256, activation = 'relu', kernel_initializer = 'he_normal')(x_sec)
    # x_sec= Dropout(0.4)(x_sec)
    x_sec= Dense(28, kernel_initializer = 'he_normal')(x_sec)
    x_sec = Reshape((28,1))(x_sec)
    x          = Reshape((28,1))(x)
    x_lest = concatenate([x,x_sec],axis=-1)
    x_lest = Lambda(lambda x : sigmoid(x))(x_lest)
    
    model = Model(inputs = base_model.input, outputs = x_lest)
    return model
    
# model = ResNet()  
# model.summary(line_length=120)