import os
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
import numpy as np
from sklearn.utils import shuffle
import time
import cv2
import tqdm
from PIL import Image
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten, Dropout
from keras.layers import Input, merge
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt
from keras.datasets import mnist
import keras.backend as K
from keras.initializers import RandomNormal
K.set_image_dim_ordering('tf')

np.random.seed(36)

def gen_1(noise_shape):

    kernel_init = 'glorot_uniform'    
    Inp = Input(shape = noise_shape)
    
    x = Conv2DTranspose(filters = 512, kernel_size = (4,4), strides = (1,1), padding = "valid", data_format = "channels_last", kernel_initializer = kernel_init)(Inp)
    x = BatchNormalization(momentum = 0.1)(x)
    x = LeakyReLU(0.2)(x)
        
    x = Conv2DTranspose(filters = 256, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(x)
    x = BatchNormalization(momentum = 0.1)(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2DTranspose(filters = 128, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(x)
    x = BatchNormalization(momentum = 0.1)(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2DTranspose(filters = 64, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(x)
    x = BatchNormalization(momentum = 0.1)(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(x)
    x = BatchNormalization(momentum = 0.1)(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2DTranspose(filters = 3, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(x)
    x = Activation('tanh')(x)
        
    g_model = Model(input = Inp, output = x)
    return g_model
    
#------------------------------------------------------------------------------------------

def disc_1(image_shape=(64,64,3)):

    kernel_init = 'glorot_uniform'
    
    dis_inp = Input(shape = image_shape)
    
    x = Conv2D(filters = 64, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(dis_inp)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2D(filters = 128, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(x)
    x = BatchNormalization(momentum = 0.1)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters = 256, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(x)
    x = BatchNormalization(momentum = 0.1)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters = 512, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(x)
    x = BatchNormalization(momentum = 0.1)(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    
    d_model = Model(input = dis_inp, output = x)
    return d_model

#------------------------------------------------------------------------------------------
