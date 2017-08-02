from sklearn.model_selection import train_test_split
from random import shuffle
import csv
import cv2
import numpy as np
import sklearn

samples = []
with open("./data/driving_log.csv") as csvFile:
    csvReder = csv.reader(csvFile)
    for line in  csvReder:
        samples.append(line)

# Split traning data
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# generator
def samples_generator(samples, batch_size=32):
    # Get total number of samples
    num_samples = len(samples)

    while True:
        shuffle(samples)
        for offset in range(0,batch_size, num_samples):
            # Get batch
            batch_samples = samples[offset:offset+batch_size]
            # Process each image in the batch
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread("./data/IMG/"+batch_sample[0].split("/")[-1])
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


