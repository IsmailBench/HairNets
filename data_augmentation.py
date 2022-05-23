
import numpy as np
import tensorflow as tf
import skimage as skimage
import os 

height, width = 224, 224
channels = 3
batch = 32

hair_segment = "datasets/hair_segment/"
hair_training = "datasets/hair_training/"

# WE SELECT THE FACES THAT WE ARE GOING TO USE
folder_images = "datasets/lfw_funneled/"
folder_mask = "datasets/parts_lfw_funneled_gt_images/"


def flip():
    i = 0
    #Path for person names
    person_name = [f.name for f in os.scandir(folder_images) if f.is_dir() ] #name of person
    person_name_segment = [f.name for f in os.scandir(hair_segment)] #name of person
    arr = np.array(person_name_segment)
    # NumPy.'img' = A single image.
    flip_1 = np.fliplr(img)
    # TensorFlow. 'x' = A placeholder for an image.
    shape = [height, width, channels]
    x = tf.placeholder(dtype = tf.float32, shape = shape)
    flip_2 = tf.image.flip_up_down(x)
    flip_3 = tf.image.flip_left_right(x)    
    flip_4 = tf.image.random_flip_up_down(x)
    flip_5 = tf.image.random_flip_left_right(x)

def rotate():
    i = 0
    #Path for person names
    person_name = [f.name for f in os.scandir(folder_images) if f.is_dir() ] #name of person
    person_name_segment = [f.name for f in os.scandir(hair_segment)] #name of person
    arr = np.array(person_name_segment)
    # Placeholders: 'x' = A single image, 'y' = A batch of images
    # 'k' denotes the number of 90 degree anticlockwise rotations
    shape = [height, width, channels]
    x = tf.placeholder(dtype = tf.float32, shape = shape)
    rot_90 = tf.image.rot90(img, k=1)
    rot_180 = tf.image.rot90(img, k=2)
    #To rotate in any angle. In the example below, 'angles' is in radians
    shape = [batch, height, width, 3]
    y = tf.placeholder(dtype = tf.float32, shape = shape)
    rot_tf_180 = tf.contrib.image.rotate(y, angles=3.1415)
    # Scikit-Image. 'angle' = Degrees. 'img' = Input Image
    # For details about 'mode', checkout the interpolation section below.
    rot = skimage.transform.rotate(img, angle=45, mode='reflect')

def scale():
    i = 0
    #Path for person names
    person_name = [f.name for f in os.scandir(folder_images) if f.is_dir() ] #name of person
    person_name_segment = [f.name for f in os.scandir(hair_segment)] #name of person
    arr = np.array(person_name_segment)
    # Scikit Image. 'img' = Input Image, 'scale' = Scale factor
    # For details about 'mode', checkout the interpolation section below.
    scale_out = skimage.transform.rescale(img, scale=2.0, mode='constant')
    scale_in = skimage.transform.rescale(img, scale=0.5, mode='constant')
    # Don't forget to crop the images back to the original size (for 
    # scale_out)

def crop():
    i = 0
    #Path for person names
    person_name = [f.name for f in os.scandir(folder_images) if f.is_dir() ] #name of person
    person_name_segment = [f.name for f in os.scandir(hair_segment)] #name of person
    arr = np.array(person_name_segment)
    # TensorFlow. 'x' = A placeholder for an image.
    original_size = [height, width, channels]
    x = tf.placeholder(dtype = tf.float32, shape = original_size)
    # Use the following commands to perform random crops
    crop_size = [new_height, new_width, channels]
    seed = np.random.randint(1234)
    x = tf.random_crop(x, size = crop_size, seed = seed)
    output = tf.images.resize_images(x, size = original_size)

def transalation():
    i = 0
    #Path for person names
    person_name = [f.name for f in os.scandir(folder_images) if f.is_dir() ] #name of person
    person_name_segment = [f.name for f in os.scandir(hair_segment)] #name of person
    arr = np.array(person_name_segment)
    # pad_left, pad_right, pad_top, pad_bottom denote the pixel 
    # displacement. Set one of them to the desired value and rest to 0
    shape = [batch, height, width, channels]
    x = tf.placeholder(dtype = tf.float32, shape = shape)
    # We use two functions to get our desired augmentation
    x = tf.image.pad_to_bounding_box(x, pad_top, pad_left, height + pad_bottom + pad_top, width + pad_right + pad_left)
    output = tf.image.crop_to_bounding_box(x, pad_bottom, pad_right, height, width)