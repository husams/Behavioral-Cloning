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
    correction = {'center': 0, 'left': 0,25, 'right': -0.25 }
    correction = [0, 0.25, -0.25]
   
    image = mpimg.imread(os.path.join(DATA_PATH, data[cameras[camera]].values[index].strip()))
    angle = data.steering.values[index] + correction[camera]

    return image, angle

def random_shift(image, steer):
    trans_range = 100
    x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + x / trans_range * 2 * .2
    y = 0
    M = np.float32([[1, 0, tr_x], [0, 1, y]])
    image_tr = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return image_tr, steer_ang

def generator(data, batch_size):
    
    # Size 
    batch_size = batch_size / 8

    while True:
        # Select n number of random items
        indices = np.random.permutation(data.count()[0])
        
        for offset in range(0, len(indices), batch_size):
            batches = indices[offset:(offset + batch_size)]

            X = []
            y = []

            # create batch
            for index in batches:
                # center 
                center_image,   center_angle   = select_image(data, index, "center")
                left_image,     left_angle     = select_image(data, index, "left")
                right_image,    right_angle    = select_image(data, index, "right")
                shiffted_image, shiffted_angle = random_shift(center_image, center_angle)

                images = [center_image, left_image, right_image, shiffted_image]
                angles = [center_angle, left_angle, right_angle, shiffted_angle] 

                for idx in range(0, 4):
                    # Random brightness
                    image = images[idx]
                    angle = angles[idx]

                    if random.sample([0,1],1)[0] == 0:
                        # Random brightness
                        image = random_brightness(image)
                    else:
                        # Random shadow
                        image = random_shadow(image)

                    # Random flip
                    image, angle = random_flip(image, angle)
                    images.append(random_brightness(image))
                    angles.append(angle)

                X.expend(images)
                y.expend(angles)

            yield sklearn.utils.shuffle(np.array(X), np.array(y))