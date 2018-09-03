# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 18:04:20 2018
@author: Carl
"""

import os
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
import glob
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from sklearn.utils import shuffle
import time
import cv2
import scipy
import imageio
from PIL import Image
import matplotlib.gridspec as gridspec
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose, Dropout
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model
from GAN_car_model import gen_1, disc_1
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.backend as K
from scipy.interpolate import spline
K.set_image_dim_ordering('tf')

from collections import deque

np.random.seed(36)

def norm_img(img):
    img = (img / 127.5) - 1
    return img

def denorm_img(img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8) 

def load_data(batch_size, image_shape, data_dir=None, data = None):
    sample_dim = (batch_size,) + image_shape
    sample = np.empty(sample_dim, dtype=np.float32)
    all_data_dirlist = list(glob.glob(data_dir))

    sample_imgs_paths = np.random.choice(all_data_dirlist,batch_size)
    for index,img_filename in enumerate(sample_imgs_paths):
        image = Image.open(img_filename)
        #print(image.size)
        #image.thumbnail(image_shape[:-1], Image.ANTIALIAS) - this maintains aspect ratio ; we dont want that - we need m x m size
        image = image.resize(image_shape[:-1])
        image = image.convert('RGB') #remove transparent ('A') layer
        #print(image.size)
        #print('\n')
        image = np.asarray(image)
        image = norm_img(image)
        sample[index,...] = image
    return sample

def save_img_batch(img_batch,img_save_dir):
    plt.figure(figsize=(4,4))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(img_batch.shape[0],16,replace=False)
    #print(rand_indices)
    for i in range(16):
        #plt.subplot(4, 4, i+1)
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        rand_index = rand_indices[i]
        image = img_batch[rand_index, :,:,:]
        fig = plt.imshow(denorm_img(image))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(img_save_dir,bbox_inches='tight',pad_inches=0)
    # plt.show()   

num_steps = 11000
batch_size = 64
###################### still need modify
img_save_dir = SCRIPT_PATH + "/train_record"
data_dir =  SCRIPT_PATH + "/images_car/*.jpg"
######################
log_dir = img_save_dir
save_model_dir = img_save_dir


# load discriminator and generator models
noise_shape = (1,1,100)
g_model = gen_1(noise_shape)
g_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
g_model.summary()
plot_model(g_model, to_file=SCRIPT_PATH+'/model_plots/generate.png')

image_shape = (64,64,3)
d_model = disc_1(image_shape)
d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
d_model.summary()
plot_model(d_model, to_file=SCRIPT_PATH+'/model_plots/discriminator.png')
d_model.trainable = False

# build gan model
gen_inp = Input(shape=noise_shape)
GAN_inp = g_model(gen_inp)
GAN_opt = d_model(GAN_inp)
gan = Model(input = gen_inp, output = GAN_opt)
gan.compile(loss = 'binary_crossentropy', optimizer = Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
gan.summary()
plot_model(gan, to_file=SCRIPT_PATH+'/model_plots/gan.png')

# loss data record
avg_disc_fake_loss = deque([0], maxlen=250)     
avg_disc_real_loss = deque([0], maxlen=250)
avg_GAN_loss = deque([0], maxlen=250)

for step in range(num_steps): 
    tot_step = step
    print("Begin step: ", tot_step)
    step_begin_time = time.time() 
    
    # load dataset
    real_data_X = load_data(batch_size, image_shape, data_dir = data_dir)

    # generate noise to picture
    noise = np.random.normal(0, 1, size=(batch_size,)+noise_shape)
    fake_data_X = g_model.predict(noise)    
    if (tot_step % 10) == 0:
        step_num = str(tot_step).zfill(4)
        save_img_batch(fake_data_X,img_save_dir + "/generateimage/" + step_num + "_image.png")

    # feed data to discriminator
    data_X = np.concatenate([real_data_X,fake_data_X])    
    real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
    fake_data_Y = np.random.random_sample(batch_size)*0.2
    data_Y = np.concatenate((real_data_Y,fake_data_Y))
    
    d_model.trainable = True
    g_model.trainable = False
    
    if step % 30 != 0:
        dis_metrics_real = d_model.train_on_batch(real_data_X,real_data_Y)   
        dis_metrics_fake = d_model.train_on_batch(fake_data_X,fake_data_Y)   
        print("Disc: real loss: %f fake loss: %f" % (dis_metrics_real[0], dis_metrics_fake[0]))
        avg_disc_fake_loss.append(dis_metrics_fake[0])
        avg_disc_real_loss.append(dis_metrics_real[0])
    else:
        dis_metrics_real = d_model.train_on_batch(fake_data_X,fake_data_Y)   
        dis_metrics_fake = d_model.train_on_batch(real_data_X,real_data_Y)  
    g_model.trainable = True
    d_model.trainable = False

    # train gan
    GAN_X = np.random.normal(0, 1, size=(batch_size,)+noise_shape)
    GAN_Y = real_data_Y
    gan_metrics = gan.train_on_batch(GAN_X,GAN_Y)
    print("GAN loss: %f" % (gan_metrics[0]))
    
    # log record
    text_file = open(log_dir+"/training_log.txt", "a")
    text_file.write("Step: %d Disc: real loss: %f fake loss: %f GAN loss: %f\n" % (tot_step, dis_metrics_real[-1], dis_metrics_fake[-1],gan_metrics[-1]))
    text_file.close()
    avg_GAN_loss.append(gan_metrics[0])
    
    end_time = time.time()
    diff_time = int(end_time - step_begin_time)
    print("Step %d completed. Time took: %s secs." % (tot_step, diff_time))
    
    if ((tot_step+1) % 500) == 0:
        print("-----------------------------------------------------------------")
        print("Average Disc_fake loss: %f" % (np.mean(avg_disc_fake_loss)))    
        print("Average Disc_real loss: %f" % (np.mean(avg_disc_real_loss)))    
        print("Average GAN loss: %f" % (np.mean(avg_GAN_loss)))
        print("-----------------------------------------------------------------")
        d_model.trainable = True
        g_model.trainable = True
        g_model.save(save_model_dir+'/models_set/'+str(tot_step)+"_GENERATOR_weights_and_arch.hdf5")
        d_model.save(save_model_dir+'/models_set/'+str(tot_step)+"_DISCRIMINATOR_weights_and_arch.hdf5")


#generator = load_model(save_model_dir+'9999_GENERATOR_weights_and_arch.hdf5')

#generate final sample images
# for i in range(10):
#     noise = np.random.normal(0, 1, size=(batch_size,)+noise_shape)
#     fake_data_X = generator.predict(noise)    
#     save_img_batch(fake_data_X,img_save_dir+"/generateimage/"+"final"+""str(i)+"_image.png")


# """
# #Display Training images sample
# save_img_batch(sample_from_dataset(batch_size, image_shape, data_dir = data_dir),img_save_dir+"_12TRAINimage.png")
# """

# #Generating GIF from PNG
# images = []
# all_data_dirlist = list(glob.glob(img_save_dir+"*_image.png"))
# for filename in all_data_dirlist:
#     img_num = filename.split('\\')[-1][0:-10]
#     if (int(img_num) % 100) == 0:
#         images.append(imageio.imread(filename))
# imageio.mimsave(img_save_dir+'movie.gif', images) 
    
# """
# Alternate way to convert PNG to GIF (ImageMagick):
#     >convert -delay 10 -loop 0 *_image.png animated.gif
# """