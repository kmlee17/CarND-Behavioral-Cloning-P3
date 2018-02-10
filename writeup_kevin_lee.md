## **Behavioral Cloning** 
## Kevin Lee Writeup

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_imgs/nvidia.png "Model Visualization"
[image2]: ./writeup_imgs/crop.png "Cropping Example"
[image3]: ./writeup_imgs/angles.png "Angles Histogram"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* track1.mp4 video of autonomous lap on track using model.h5
* writeup_kevin_lee.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of the network architecture from the NVidia paper.

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer between the convolutional and fully connected layers in order to reduce overfitting (model.py lines 94).

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 64). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 105).  I did play a little bit with the steering angle adjustment for the left/right camera images, and settled on a value of 0.21 after eyeballing how centered the car drove with different values.

#### 4. Appropriate training data

The training data I used was the data set provided by Udacity.  I did attempt to collect my own data through navigating the course manually, but I wasn't too happy with the precision using a keyboard on a laptop.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use an architecture that had been proven to work well in self-driving car situations before.  That is why I chose the NVidia model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that there was a dropout layer in between the convolutional layers and the fully connected layers.

I first started with just using the center images and saw that the model sometimes failed to recognize turns.  I then tried to limit some of the external noise in the image by cropping it so that just the road portion was visible rather than the background environment.

![alt text][image2]

I then found that the vehicle was having issues with some of the turns.  I then decided to add in the left and right images as well to hopefully make the model more robust and better handle turns/edge steering.  I added in a steering angle adjustment value for the left and right images as well as these images are offset.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Lambda         		| 160x320x3 RGB image   					| 
| Cropping   	| outputs 80x320x3 	|
| Convolution 	| 5x5 filter, 24 features, 2x2 stride, ELU 			|
| Convolution 	| 5x5 filter, 36 features, 2x2 stride, ELU 			|
| Convolution 	| 5x5 filter, 48 features, 2x2 stride, ELU 			|
| Convolution 	| 3x3 filter, 64 features, 1x1 stride, ELU 			|
| Convolution 	| 3x3 filter, 64 features, 1x1 stride, ELU 			|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, outputs 10x10x16      		|
| Dropout		| 0.5 probability     									|
| Flatten		| 
| Fully connected		| outputs 100, ELU                          |
| Fully connected		| outputs 50, ELU                          |
| Fully connected		| outputs 10, ELU                          |
| Fully connected		| outputs 1, ELU                          |

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I had 8036 number of data points which I split into a training data set and a validation data set (80/20 split). I then preprocessed this data by converting images back to the RGB color space using OpenCV because of the note that the drive.py loads images in this space.  After the color space conversion, I normalized the images so that the model can avoid dealing with scale differences.  This will help the training process with more stable gradients.

I plotted a histogram of the steering angles to see what the output distribution looked like and found it to be heavily weighted towards a 0.0 steering angle.

![alt text][image3]

This makes sense as the car is usually going straight.  However, I thought this distribution could be problematic.  In the end, the car navigated the track fine even with this unbalanced distribution, however, it may not hold up as well on a completely new track.  Data augmentation to balance out the observations could help here.

I first tried using the standard fit method, but found that the images took a very long time to process.  I then switched to using a generator to feed into the fit_generator function.  This, along with training the network on a GPU-enabled computer helped train time dramatically.

I used the training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by trial and error where I ran a larger number of epochs and found that the minimum loss occured around the 5th epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
