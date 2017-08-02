import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Conv2D, MaxPool2D, Activation, Dropout
from keras.optimizers import Adam
from preprocess import train_samples, validation_samples, samples_generator

print("Start traning")
# Create simple model

# compile and train the model using the generator function
train_generator = samples_generator(train_samples, batch_size=32)
validation_generator = samples_generator(validation_samples, batch_size=32)

opt = Adam(lr=0.0001)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(32, (5,5), padding="same"))
model.add(Activation('relu'))
model.add(Conv2D(64, (5,5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (5,5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss="mse", optimizer=opt, metrics=['accuracy'])
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), nb_val_samples=len(validation_samples), nb_epoch=10)

model.save("model.h5")

import gc; gc.collect()
