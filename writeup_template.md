# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/ConvNet.png "Model Visualization"
[image2]: ./images/samples.png "Samples"
[image3]: ./images/cameras.png "Recovery Image"
[image4]: ./images/flipped.png "Flipped Image"
[image5]: ./images/shiffted.png "Shiffted"
[image6]: ./images/brightness.png "Brightness"
[image7]: ./images/shadow.png "shadow"
[image8]: ./examples/placeholder_small.png "Flipped Image"
[image9]: ./images/distribution.png "Data Distribution"
[image10]: ./images/preprocessed.png "Preprocessed"


# Files Submitted & Code Quality


### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* data.py for data Augmentation.
* preprocess.py image pre-process pipeline.
* model.h5 containing a trained convolution neural network 
* run.mp4 movies shows the car driving round track 1 autonomously. 
* writeup_report.md or writeup_report.pdf summarizing the results

### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model.

The data.py file contains the code for the generator which used in traning and validation. The module also contains set of utility function used to perform different type of image transformation.

The preprocess.py file contaons function which used to resize the image and change colorspace. the module used in  model.py and drive.py.

All of the above files contains comments to explain how the code works.

# Model Architecture and Training Strategy

### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 16, 32 and 64, and Max pooling with stride of 2 between each convolutional layer.

The model includes two fully connected layers with RELU activation to introduce nonlinearity  followed by single liner layer with one unit for regression, and the data is normalized in the model using a Keras lambda layer.

### 2. Attempts to reduce overfitting in the model

The following steps was taken to reduce overfitting
1. Dropout layer betwoeen one of the connected layers.
2. Using data augmentation to generate more samples.
3. Traning and validating model with different data sets. 


### 3. Model parameter tuning

The model used an adam optimizer with the following paramters:

1. Learning rate 0.0001 which was good enought to train the model
2. Batch size 32
3. Epochs 3

I experimented with different number of  batch sizes and epochs and I had better result with batch size 32 and 3 epochs.


### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.
I also used augmentation to increase number of sample and blanch the data set.

For details about how I created the training data, see the next section. 

# Model Architecture and Training Strategy

### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to lower the mean squared error for traning and validation and predict steering angle which allow the model to keep the car inside the track.

My first step was to use a convolution neural network model similar to the Nvidia I thought this model might be appropriate because it was designed for simiker task.

THe inital traning shows the model was overfitting because the capacity was to large for the dataset I was using, so I decided first o impove the data set before make any changes in model. 

Second step was to train the model using the approved data set shows impovment but still overfitting, so I decide to add dropout layers seems to have good impact. 

 At this stage the model was able to predict  with acceptable loss without overfitting, However I dcieded to make further changes to the model to reduce traning and validation loss and memory footprint.

To reduce the losss I introduced max pooling with stride 2 between convolutional layers. And to reduce memory footprint I decided to reduce number of units in the fully connected layer and use only three convolutional layers instead of four. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes 


| Layer (type)         |        Output Shape       |        Param #  |
|:--------------------:|:-------------------------:|:---------------:|
| Lambda               |        3@64 x 64          |       0         |
|                      |                           |                 |
| Convolutional        |      16@62 x 62           |       448       |
|                      |                           |                 |
| Max Pooling 2x2      |       16x@31 x 31         |        0        |
|                      |                           |                 |
| Convolutional        |      32@29 x 29           |       4640      |
|                      |                           |                 |
| Max Pooling 2x2      |      32@14 x 14           |       0         |
|                      |                           |                 |
| Convolutional        |      64@12 x 12           |       18496     |
|                      |                           |                 |
|Max Pooling 2x2       |       64@6 x 6            |       0         |
|                      |                           |                 |
|Flatten               |      2304                 |       0         |
|                      |                           |                 |
|Fully Connected       |       512                 |       1180160   |
|                      |                           |                 |
|Dropout               |       512                 |       0         |
|                      |                           |                 |
|Fully Connected       |       64                  |       32832     |
|                      |                           |                 |
|Fully Connected       |       16                  |       1040      |
|                      |                           |                 |
|Fully Connected       |       1                   |       17        |


|                      |              |
|:--------------------:|-------------:|
| Total params         |  1,237,633.0 |
| Trainable params     |  1,237,633.0 |
| Non-trainable params |  0.0         |


Here is a visualization of the architecture

![alt text][image1]

### 3. Creation of the Training Set & Training Process

Because it was defcult for me to steer the car with he keyboard,  I decided  to use the  dataset provided by Udacity instead of the generating new one. 

The dataset was mostly centered around zero steering angle as sowing in below figure which means most likliy model will not be able to steer car back to the center of the track.

![alt text][image9]

And here is some examples from the dataset

![alt text][image2]

In order to compat overfitting and improve traning so that the model would learn how to steer the vehicle back to the center, I decided to use the following  techniques:


### * Use Left and Right cameras:

I made use of the images for  left and right cameras, by adding 0.25 to angle for the left image and remove 0.25 from  the angle for the right image as showing in below images:

![alt text][image3]

### * Flipping:

In this technique the camera image was mirred and steering angle was reversed, so the model can learn to steer when the car in the opposite position.

![alt text][image4]

### * Horizontal and Vertical Shifts

In this technique random horizontal and vertical shift was applied so the model can steer when the car in different position.

![alt text][image5]

### * Change brightness:

In this technique camera image brightness was changed by converting the image to HSV color space and  randomly scaling V channel to simulate different lighting condition 

![alt text][image6]

### * Drop Shadow:

In this technique a randomly select region of camera image brightness was changed to simulate shadow effect.


![alt text][image7]

Preprocessing was applied also to all images which consist of:
1. Resizing the image down to 64x64
2. Xhange color space to YUV
3. Cropping to remove unwantted region.

![alt text][image10]

The dataset was randomly shuffled and split into training, validation and test dataset where 19% was used for validation and 1% for test to evalute the model. python generator was used to augment and create random samples for traning and generate samples witout augmentation for the validation.


The model was finally tranined with 45K traning samples, and 1447 for validation which gave the following result

|                |        |
|:--------------:|:------:|
|Traing loss     | 0.0262 |
|Validation loss | 0.0238 |
|Test loss       | 0.0234 |

The model was able to drive the car round track 1 autonomously without leavung the track.