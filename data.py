from sklearn.model_selection import train_test_split
from random import shuffle
import skimage.transform as transform
import csv
import cv2
import numpy as np
import sklearn
import random
import matplotlib.image as mpimg
import sklearn.utils
import os
from numpy import newaxis
import math

DATA_PATH = "./data"

def normalize(x, min=-0.5, max=0.5):
    x_min = np.min(x)
    x_max = np.max(x)
    return ((x - x_min) / (x_max - x_min)) * (max-min) + min

def random_brightness(image):
    # Randomly select a percent change
    change_pct = random.uniform(0.4, 1.2)
    
    # Change to HSV to change the brightness V
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * change_pct
    
    #Convert back to RGB 
    img_brightness = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img_brightness

def random_shadow(image):
    h, w = image.shape[0], image.shape[1]
    [x1, x2] = np.random.choice(w, 2, replace=False)
    k = h / (x2 - x1)
    b = - k * x1
    for i in range(h):
        c = int((i - b) / k)
        image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)
    return image

def random_flip(image, angle):
    if random.sample([0,1],1)[0] == 0:
        # flip the image 
        image = np.fliplr(image)
        angle = -angle
    return image, angle

def select_image(data, index, camera):
    correction = {'center': 0, 'left': 0.25, 'right': -0.25 }
    correction = [0, 0.25, -0.25]
   
    image = mpimg.imread(os.path.join(DATA_PATH, data[cameras[camera]].values[index].strip()))
    angle = data.steering.values[index] + correction[camera]

    return image, angle

def random_camera(data, index):
    camera     = random.sample(['center', 'left', 'right'], 1)[0]
    correction = {'center': 0, 'left': 0.25, 'right': -0.25 }
   
    image = mpimg.imread(os.path.join(DATA_PATH, data[camera].values[index].strip()))
    angle = data.steering.values[index] + correction[camera]

    return image, angle

def random_shift(image, steer):
    trans_range = 100
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(image.shape[1], image.shape[0]))
    
    return image_tr, steer_ang

def image_augmentation(image, angle):
    image        = random_brightness(image)
    image        = random_shadow(image)
    image, angle = random_shift(image, angle)
    image, angle = random_flip(image, angle)

    return image, angle

def preprocess(image):
    shape = image.shape
    #image = image[55:shape[0]-25, 0:shape[1]]
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(64,64),  interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    image = image[:,:,2] 
    image = image[..., newaxis]
    return (image/255.0-.5)


traning_set = []

def generator(data, batch_size, augmentation=True, trace=False):
    global traning_set
    
    X, y = [], []

    while True:
        data = sklearn.utils.shuffle(data)
        for  index in range(data.shape[0]):
            # Select camera randmly
            image, angle = random_camera(data, index)

            # Augmentation
            if augmentation:
                image, angle = image_augmentation(image, angle)

            # resize image
            image = preprocess(image)

            X.append(image)
            y.append(angle)

            if len(y) == batch_size:
                if trace:
                    traning_set.extend(y)
                yield sklearn.utils.shuffle(np.array(X), np.array(y))
                X, y = [], []