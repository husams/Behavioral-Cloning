import csv
import cv2
import numpy as np

import keras.backend as K
from keras.callbacks import TensorBoard
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
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(66,200,3)))
model.add(Convolution2D(24, kernel_size=(5,5), strides=(2, 2), padding='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(36, kernel_size=(5,5), strides=(2, 2), padding='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(48, kernel_size=(5,5), strides=(2, 2), padding='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, kernel_size=(3,3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, kernel_size=(3,3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1164,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(100,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(50,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss="mse", optimizer=opt, metrics=['accuracy'])
model.summary()

K.set_learning_phase(1)
K.set_image_data_format('channels_last')

tensorBoard = TensorBoard(
    log_dir='./logs',
    histogram_freq=2,
    write_graph=True,
    write_images=False)
tensorBoard.set_model(model)

model.fit_generator(
    train_generator, 
    steps_per_epoch=len(train_samples), 
    validation_steps=len(validation_samples), 
    nb_epoch=5,
    verbose=1,
    callbacks=[tensorBoard])

model.save("model.h5")

import gc; gc.collect()
