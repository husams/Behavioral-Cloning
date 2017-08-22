from sklearn.model_selection import train_test_split
from random import shuffle
from preprocess import preprocess
import cv2
import numpy as np
import sklearn
import random
import matplotlib.image as mpimg
import sklearn.utils
import pandas as pd
import os
import math

DATA_PATH = "./data"

def random_brightness(image):
    """
        Randomly chamge the brightness of an image
        1. Randomly select brightness level
        2. Change color space from RGB to HSV
        3. Change brightness by changing values for S channel
        4. Change back to RGB color space
    """
    change_pct = random.uniform(0.4, 1.2)  
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * change_pct
    img_brightness = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img_brightness

def random_shadow(image):
    """
        Add random  shadow to the image.
        1.  Randomly select to two point in the x-axis
        2. compute the height of the region
        3. change the brightnes of the select region
    """
    h, w = image.shape[0], image.shape[1]
    [x1, x2] = np.random.choice(w, 2, replace=False)
    k = h / (x2 - x1)
    b = - k * x1
    for i in range(h):
        c = int((i - b) / k)
        image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)
    return image

def random_flip(image, angle):
    """
        Randomly filp the image
    """
    if random.sample([0,1],1)[0] == 0:
        # flip the image 
        image = np.fliplr(image)
        angle = -angle
    return image, angle

def center_image(data, index):
    """
        Load center image, this using during
        testing and validation.
    """
    image = mpimg.imread(os.path.join(DATA_PATH, data['center'].values[index].strip()))
    angle = data.steering.values[index]
    return image, angle

def random_camera(data, index):
    """
        Randomly select camera image.
        1. Select camera randomly
        2. Select the angle correction based on the camera
        3. Load the image and compute new angle
    """
    camera     = random.sample(['center', 'left', 'right'], 1)[0]
    correction = {'center': 0, 'left': 0.25, 'right': -0.25 }
   
    image = mpimg.imread(os.path.join(DATA_PATH, data[camera].values[index].strip()))
    angle = data.steering.values[index] + correction[camera]

    return image, angle

def random_shift(image, steer):
    """
        Randomly apply virtical and horizontal shift
        1. Select random horizontal translation
        2. Compute the new steering angle using select translation
        3. Apply horizontal translation
        4. elect random virtical translation
        5. Apply virtical translation
    """
    trans_range = 100
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(image.shape[1], image.shape[0]))
    
    return image_tr, steer_ang

def image_augmentation(image, angle):
    """
       Augmente image by 
       1. Changing brightness
       2. Drop shadow
       3. virtical and horizontal
       4. fliping
    """
    image        = random_brightness(image)
    image        = random_shadow(image)
    image, angle = random_shift(image, angle)
    image, angle = random_flip(image, angle)

    return image, angle

def generator(data, batch_size, training=True):
    """
        Generate samples  for traning/validation/test
    """
    X, y = [], []

    while True:
        data = sklearn.utils.shuffle(data)
        for  index in range(data.shape[0]):
            if training:
                # Select camera randmly
                image, angle = random_camera(data, index)
            else:
                # Use center image during validation and test
                image, angle  = center_image(data, index)

            # Augmentation only during training
            if training:
                image, angle = image_augmentation(image, angle)

            # resize image
            image = preprocess(image)

            X.append(image)
            y.append(angle)

            if len(y) == batch_size:
                # Shuffle and generate batch
                yield sklearn.utils.shuffle(np.array(X), np.array(y))
                X, y = [], []
                
# Load data
data = pd.read_csv('./data/driving_log.csv')
# shuffle data
data = sklearn.utils.shuffle(data)

# Create training / test / validation set
train_samples, test_samples = train_test_split(data, test_size=0.2)
validation_samples, test_samples = train_test_split(test_samples, test_size=0.1)
