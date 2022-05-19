# -*- coding: utf-8 -*-

#from email.mime import image
import os
import numpy as np
import cv2
import requests
import sys, tarfile
import shutil

# WE SELECT THE FACES THAT WE ARE GOING TO USE
folder_images = "datasets/lfw_funneled/"
folder_mask = "datasets/parts_lfw_funneled_gt_images/"

hair_segment = "datasets/hair_segment/"
hair_training = "datasets/hair_training/"


def extract(tar_url, extract_path='.'):
    tar = tarfile.open(tar_url, 'r')
    print("Processing extract")
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract(item.name, "./" + item.name[:item.name.rfind('/')])
try:

    extract(sys.argv[1] + '.tgz')
    print('Done.')
except:
    name = os.path.basename(sys.argv[0])
    print(name[:name.rfind('.')], '<filename>')

def copy_faces():
#This function selects only images which ground truth masks labels are provided in the LBW dataset
    i = 0
    #Path for person names
    person_name = [f.name for f in os.scandir(folder_images) if f.is_dir() ] #name of person
    person_name_segment = [f.name for f in os.scandir(hair_segment)] #name of person
    arr = np.array(person_name_segment)
    for person in person_name:
        #path for images on .dat for that person
        path_dat =  folder_images + person + "/"
        #list of all dat images for that person
        images_dat = os.listdir(path_dat)
    
        for images in images_dat:
            if images in arr:
                img = cv2.imread(folder_images + person +"/"+images)
                cv2.imwrite(hair_training+images,img)
                i = i+1

    print("Total number of label faces in the wild with ground truth ", i)

def rgbToGray():
#This function selects only images which ground truth masks labels are provided in the LBW dataset
    i = 0
    #Path for person names
    person_name = [f.name for f in os.scandir(folder_gt_images) if f.is_dir() ] #name of person
    person_name_segment = [f.name for f in os.scandir(hair_segment)] #name of person
    arr = np.array(person_name_segment)
    for person in person_name:
        #path for images on .dat for that person
        path_dat =  "datasets/lfw_funneled/" + person + "/"
        #list of all dat images for that person
        images_dat = os.listdir(path_dat)
    
        for images in images_dat:
            if images in arr:
                img = cv2.imread("datasets/lfw_funneled/"   + images)
                rgb_weights = [0.2989, 0.5870, 0.1140]
                img = np.dot(img[...,:3], rgb_weights)
                cv2.imwrite("datasets/hair_training/"+images,img)
                i = i+1

    print("Total number of label faces in the wild with ground truth ", i)

def convert_mask_gray():
    
    #MASK ARE CONVERTED TO binary mask
    #WE convert PPM to JPG
    #We only take the red channel that correspons to hair segment that is number 2
    #Images and Masks are resized to 128x128

    allmasks = os.listdir(folder_mask)

    width,height = 224, 224
    dim = (width, height)

    for mask in allmasks:
        img = cv2.imread(folder_mask+mask)
        image_name = mask.replace(".ppm", "")
        image_name = image_name.replace("._", "")
        image_jpg = image_name + ".jpg" 
    # Face image in grayscale
        if (img is not None):
        #Hair segment
            segment = img[:, :, 2]   # the hair segment is the red channel
            resized = cv2.resize(segment, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(hair_segment+image_jpg,resized)

######################################## Creating Folder #########################################################

shutil.rmtree("datasets/")
os.mkdir("datasets/")
os.mkdir(hair_segment)
os.mkdir(hair_training)

######################################### Downloading Datasets ###################################################

url = "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"
# Downloading the file by sending the request to the URL
req = requests.get(url)
# Split URL to get the file name
filename = url.split('/')[-1]
 
# Writing the file to the local file system
with open(filename,'wb') as output_file:
    output_file.write(req.content)
print('Downloading Completed')
extract(filename, "datasets/")
os.remove(filename)

url = "http://vis-www.cs.umass.edu/lfw/part_labels/parts_lfw_funneled_gt_images.tgz"
req = requests.get(url)
# Split URL to get the file name
filename = url.split('/')[-1]
 
# Writing the file to the local file system
with open(filename,'wb') as output_file:
    output_file.write(req.content)
print('Downloading Completed')
extract(filename, "datasets/")
os.remove(filename) 


######################################## Transform Datasets ########################################################

#rgbToGray()
convert_mask_gray()
copy_faces()