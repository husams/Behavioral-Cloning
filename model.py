import csv
import cv2
import numpy as np
import math

import matplotlib.pyplot as plt
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
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(24, kernel_size=(5,5), strides=(2, 2), padding='valid', activation="relu"))
model.add(Convolution2D(36, kernel_size=(5,5), strides=(2, 2), padding='valid', activation="relu"))
model.add(Convolution2D(48, kernel_size=(5,5), strides=(2, 2), padding='valid', activation="relu"))
model.add(Convolution2D(64, kernel_size=(3,3), strides=(1, 1), padding='valid', activation="relu"))
model.add(Convolution2D(64, kernel_size=(3,3), strides=(1, 1), padding='valid', activation="relu"))
model.add(Flatten())
model.add(Dense(1164,activation="relu"))
#model.add(Dropout(0.5))
model.add(Dense(100,activation="relu"))
#model.add(Dropout(0.5))
model.add(Dense(50,activation="relu"))
#model.add(Dropout(0.5))
model.add(Dense(10, activation="relu"))
#model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss="mse", optimizer=opt)
model.summary()

batch_size       = 32
steps_per_epoch  = math.ceil(len(train_sample * 3)/batch_size)
validation_steps = math.ceil(len(validation_samples * 3)/32)

history = model.fit_generator(
    train_generator, 
    steps_per_epoch=steps_per_epoch, 
    validation_steps=validation_steps, 
    validation_data=validation_generator,
    epochs=5,
    verbose=1).history

## plot the training and validation loss for each epoch
#plt.plot(history['loss'])
#plt.plot(xhistory['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()

model.save("model.h5")

import gc; gc.collect()
