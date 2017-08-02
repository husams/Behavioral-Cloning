import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, MaxPool2D, Activation, Dropout,Cropping2D, Convolution2D
from keras.optimizers import Adam
from preprocess import train_samples, validation_samples, samples_generator

print("Start traning")
# Create simple model

# compile and train the model using the generator function
train_generator = samples_generator(train_samples, batch_size=32)
validation_generator = samples_generator(validation_samples, batch_size=32)

opt = Adam(lr=0.0001)

model = Sequential()
#model.add(Cropping2D(cropping=((25,10), (0,0)), input_shape=(66,200,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24, kernel_size=(5,5), strides=(2, 2), padding='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(36, kernel_size=(5,5), strides=(2, 2), padding='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(48, kernel_size=(5,5), strides=(2, 2), padding='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, kernel_size=(3,3), strides=(2, 2), padding='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, kernel_size=(3,3), strides=(2, 2), padding='valid'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss="mse", optimizer=opt, metrics=['accuracy'])
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), nb_val_samples=len(validation_samples), nb_epoch=10)

model.save("model.h5")

import gc; gc.collect()
