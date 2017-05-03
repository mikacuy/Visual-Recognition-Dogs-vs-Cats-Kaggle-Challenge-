import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import os
import glob
import re
import cv2


TRAIN_DIR = '/home/mikacuy/dogs_vs_cats/train/'

IMAGE_SIZE = 224; 
CHANNELS = 3
pixel_depth = 255.0  

train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) 
    if (img.shape[0] >= img.shape[1]): # height is greater than width
       resizeto = (IMAGE_SIZE, int (round (IMAGE_SIZE * (float (img.shape[1])  / img.shape[0]))));
    else:
       resizeto = (int (round (IMAGE_SIZE * (float (img.shape[0])  / img.shape[1]))), IMAGE_SIZE);
    
    img2 = cv2.resize(img, (resizeto[1], resizeto[0]), interpolation=cv2.INTER_CUBIC)
    img3 = cv2.copyMakeBorder(img2, 0, IMAGE_SIZE - img2.shape[0], 0, IMAGE_SIZE - img2.shape[1], cv2.BORDER_CONSTANT, 0)
        
    return img3[:,:,::-1]  # turn into rgb format

def prep_data(image_file):
    image = read_image(image_file);
    cv2.imshow("sample_image1", image)

    image_data = np.array (image, dtype=np.float32)
    image_data[:,:,0] -= np.mean(image_data[:,:,0])
    image_data[:,:,0] = (image_data[:,:,0].astype(float)) / pixel_depth
    image_data[:,:,1] -= np.mean(image_data[:,:,1])
    image_data[:,:,1] = (image_data[:,:,1].astype(float)) / pixel_depth
    image_data[:,:,2] -= np.mean(image_data[:,:,2])
    image_data[:,:,2] = (image_data[:,:,2].astype(float)) / pixel_depth


    cv2.imshow("sample_image2", image_data)
    cv2.waitKey()

prep_data(train_dogs[1])