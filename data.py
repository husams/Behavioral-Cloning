from sklearn.model_selection import train_test_split
from random import shuffle
import skimage.transform as transform
import csv
import cv2
import numpy as np
import sklearn
import random
import matplotlib.image as mpimg
import os

DATA_PATH = "./data"

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
    x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + x / trans_range * 2 * .2
    y = 0
    M = np.float32([[1, 0,  x], [0, 1, y]])
    image_tr = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return image_tr, steer_ang

def image_augmentation(image, angle):
    image        = random_brightness(image)
    image, angle = random_shift(image, angle)
    image, angle = random_flip(image, angle)

    return image, angle

def preprocess(image):
    image = image[60:-22,:,:]
    return transform.resize(image, (66,200), mode='constant')

def generator(data, batch_size, augmentation=True):
    X, y = [], []

    while True:
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
                yield sklearn.utils.shuffle(np.array(X), np.array(y))
                X, y = [], []