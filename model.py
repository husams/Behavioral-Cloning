import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Conv2D, MaxPool2D, Activation, Dropout
from keras.optimizers import Adam

def load_from_csv(path):
    lines = []
    with open(path +"/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    
    images = []
    angles = []
    for line in lines:
        # Extract image file name
        filename  = line[0].split('/')[-1]
        imagefile = path + "/IMG/"+ filename
        # Read image
        image = cv2.imread(imagefile)
        images.append(image)
        angle  = float(line[3])
        angles.append(angle)

    return np.array(images), np.array(angles)

# load data
print("Loading data ... ")
x_train, y_train = load_from_csv("./data")

print("Start traning")
# Create simple model

opt = Adam()

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(16, (5,5), padding="same"))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(32, (5,5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5,5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model,add(Dense(1024))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss="mse", optimizer=opt, metrics=['accuracy'])
model.fit(x_train, y_train, validation_split=0.2, shuffle=True, verbose=1)

model.save("model.h5")

import gc; gc.collect()
