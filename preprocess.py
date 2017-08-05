from sklearn.model_selection import train_test_split
from random import shuffle
import skimage.transform as transform
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
            # CSV data center,left,right,steering,throttle,brake,speed
            # Get batch
            batch_samples = samples[offset:offset+batch_size]
            # Process each image in the batch
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread("./data/IMG/"+batch_sample[0].split("/")[-1])
                left_image   = cv2.imread("./data/IMG/"+batch_sample[1].split("/")[-1])
                right_image  = cv2.imread("./data/IMG/"+batch_sample[2].split("/")[-1])

                #center_image = transform.resize(center_image, (66,200))

                correction   = 0.2
                center_angle = float(batch_sample[3])
                left_angle   = center_angle + correction
                right_angle  = center_angle - correction
                images.extend([center_image,left_image,right_image])
                angles.extend([center_angle, left_angle, right_angle])

                #center_image_flipped = np.fliplr(center_image)
                #center_angle_flipped = -center_angle
                #images.append(center_angle_flipped)
                #angles.append(center_angle_flipped)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


