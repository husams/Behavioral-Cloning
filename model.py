from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Activation, Dropout, Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from data import generator, train_samples, validation_samples, test_samples
import math

def samples_count(total_count, batch_size):
    """
      Compute number of ssample per epoch
    """
    return math.ceil(total_count/batch_size) * batch_size

learning_rate = 0.0001
batch_size    = 32
epochs        = 3

# Create generators
train_generator      = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size, training=False)
test_generetaor      = generator(test_samples, batch_size=test_samples.shape[0], training=False)

def build_model(learning_rate=0.0001):
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(64,64,3)))
    model.add(Convolution2D(16, 3,3, subsample=(1, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(32, 3,3, subsample=(1, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(64, 3,3, subsample=(1, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(512,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1))
    
    opt = Adam(lr=learning_rate)
    model.compile(loss="mse", optimizer=opt)
    model.summary()
    
    return model

# Build model
model = build_model()


# Compute number of samples / epoch
samples_per_epoch  = samples_count(45000, batch_size)
nb_val_samples     = samples_count(validation_samples.shape[0], batch_size)


# Start training
print("Start traning")
history = model.fit_generator(
    train_generator, 
    samples_per_epoch=samples_per_epoch, 
    validation_data=validation_generator,
    nb_val_samples=nb_val_samples, 
    nb_epoch=epochs,
    verbose=1).history

# Create test data
X,y = next(test_generetaor)

# Evaluate model
loss = model.evaluate(X,y)
print("Test loss : {0:.4f}".format(loss))

# Save model
model.save("model.h5")

#import gc; gc.collect()
