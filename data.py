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


def random_flip(image, angle):
    if np.random.randint(0, 1) == 0:
        # flip the image 
        image = np.fliplr(image)
        angle = -angle
    return image, angle

def select_image(data, index):
    cameras    = ['center', 'left', 'right']
    correction = [0, 0.25, -0.25]
    camera     = np.random.randint(0,3) 

    image = mpimg.imread(os.path.join(DATA_PATH, data[cameras[camera]].values[index].strip()))
    angle = data.steering.values[index] + correction[camera]

    return image, angle

def generator(data, batch_size):
    while True:
        # Select n number of random items
        batch = data.sample(batch_size)

        X = []
        y = []

        # create batch
        for index in range(0, batch_size):
            # randmly select camera image
            image, angle = select_image(data, index)
            # flip image only if angle is > 0
            if angle > 0:
                image, angle = random_flip(image, angle)
            # random brightness
            image = random_brightness(image)

            X.append(image)
            y.append(angle)
        yield np.array(X), np.array(y)

