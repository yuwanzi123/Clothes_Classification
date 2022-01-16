import numpy as np
from scipy import ndimage
#import scipy.ndimage
import imageio
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os, sys

path = "/home/zz/Closet/database/raw_data/google_tshirt/"
new_path = "/home/zz/Closet/database/data/c5_tshirt/"
dirs = os.listdir(path)

#4 flip + rotation
data_hf = ImageDataGenerator(zoom_range = [1.0, 1.5],
                             rotation_range = 45)
for item in dirs:
    image = imageio.imread(path + item)
    image = image.reshape((1, ) + image.shape)
    i = 0
    for batch in data_hf.flow(image,
                              batch_size = 32,
                              save_to_dir = new_path,
                              save_prefix = 'skirt',
                              save_format = 'jpg'):
        i += 1
        if i > 0:
            break


#? zoom
data_hf = ImageDataGenerator(zoom_range = [1.25, 1.5])
for item in dirs:
    image = imageio.imread(path + item)
    image = image.reshape((1, ) + image.shape)
    i = 0
    for batch in data_hf.flow(image,
                              batch_size = 32,
                              save_to_dir = new_path,
                              save_prefix = 'skirt',
                              save_format = 'jpg'):
        i += 1
        if i > 0:
            break



'''
#1 roataion + brightness
data_r = ImageDataGenerator(rotation_range = 30,
                            brightness_range = [0.5, 1.5])
for item in dirs:
    image = imageio.imread(path + item)
    image = image.reshape((1, ) + image.shape)
    i = 0
    for batch in data_r.flow(image,
                              batch_size = 32,
                              save_to_dir = new_path,
                              save_prefix = 'ran',
                              save_format = 'jpg'):
        i += 1
        if i > 0:
            break
'''


#2 flip
'''
data_hf = ImageDataGenerator(horizontal_flip = True)
for item in dirs:
    image = imageio.imread(path + item)
    image = image.reshape((1, ) + image.shape)
    i = 0
    for batch in data_hf.flow(image,
                              batch_size = 32,
                              save_to_dir = new_path,
                              save_prefix = 'pants',
                              save_format = 'jpg'):
        i += 1
        if i > 1:
            break
'''

'''
#3 brightness
data_br = ImageDataGenerator(brightness_range = [0.5, 1.5])
for item in dirs:
    image = imageio.imread(path + item)
    image = image.reshape((1, ) + image.shape)
    i = 0
    for batch in data_br.flow(image,
                              batch_size = 32,
                              save_to_dir = new_path,
                              save_prefix = 'slippers',
                              save_format = 'jpg'):
        i += 1
        if i > 1:
            break

'''


#5 shift
'''
data_ws = ImageDataGenerator(width_shift_range = 0.2,
                             height_shift_range = 0.2)
for item in dirs:
    image = imageio.imread(path + item)
    image = image.reshape((1, ) + image.shape)
    i = 0
    for batch in data_ws.flow(image,
                              batch_size = 32,
                              save_to_dir = new_path,
                              save_prefix = 'slippers',
                              save_format = 'jpg'):
        i += 1
        if i > 1:
            break
'''

#6 shift
'''
data_hs = ImageDataGenerator(height_shift_range = 0.2)
for item in dirs:
    image = imageio.imread(path + item)
    image = image.reshape((1, ) + image.shape)
    i = 0
    for batch in data_ws.flow(image,
                              batch_size = 32,
                              save_to_dir = new_path,
                              save_prefix = 'slippers',
                              save_format = 'jpg'):
        i += 1
        if i > 1:
            break
'''


'''
data_zr = ImageDataGenerator(zoom_range = [0.5, 1.0])
for item in dirs:
    image = imageio.imread(path + item)
    image = image.reshape((1, ) + image.shape)
    i = 0
    for batch in data_zr.flow(image,
                              batch_size = 32,
                              save_to_dir = new_path,
                              save_prefix = 'slippers',
                              save_format = 'jpg'):
        i += 1
        if i > 1:
            break
'''

