import cv2
import numpy as np
import math

def preprocess(image):
    """
        Preprocess teh image
        1. Crop the image
        2. Chage the size to 6x64
        3.Change color space to YUV
    """
    shape = image.shape
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(64,64),  interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image
