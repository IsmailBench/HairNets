# -*- coding: utf-8 -*-
import os
import gc
gc.collect()

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#%matplotlib inline

from keras.layers import Input
import sys
sys.path.insert(0, 'libs/')
from curliqfunctions import plot_sample, load_type_images, hair_extract
from curliqfunctions import save_hair_segment
from curliqnet import get_unet

print('Packet imported successfully')

# Set some parameters
im_width = 224
im_height = 224
number_channel = 3
threshold_hair = 0.50 #Threshold for binarization

#Location of images
image_folder = "datasets/test/"

# Reading the images rgb
X_gray, X_rgb, X_name = load_type_images(image_folder,image_folder)

print('Type A is ', np.shape(X_gray))

print('starting with Unet')
##### Convolutional Neural Network For Hair Segmentation
input_img = Input((im_height, im_width, 1), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model.load_weights('weights/weights.h5')

# Predict hair segment
preds_hair_segment = model.predict(X_gray, verbose=1)
#print(preds_hair_segment)

# Threshold for binary hair segment
preds_hair_segment_binay = (preds_hair_segment > threshold_hair)

print("Prediction Finished")

# Extraction of pixels corresponding to hao
X_rgb_segment = hair_extract(X_rgb, preds_hair_segment_binay)

# Save hair segment
folder_save = "datasets/results/"

save_hair_segment(X_rgb_segment, X_name, folder_save)

plot_sample(X_rgb, preds_hair_segment, X_rgb_segment, preds_hair_segment_binay)















