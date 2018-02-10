# P3 - Behavioral Cloning
# Kevin Lee

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.layers import Conv2D, Cropping2D, Dense, Flatten, Dropout, Lambda
from keras.models import Sequential

# Set global parameters
steering_adjustment = 0.21
data_path = 'data/'
batch_size = 64
epochs = 5

# Generator for fit function
def generator(samples, batch_size=64):
	num_samples = len(samples)
	while 1:
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images_center = []
			images_left = []
			images_right = []
			angles_center = []
			angles_left = []
			angles_right = []

			for obs in batch_samples:
				img_path_center = data_path + obs[0].strip()
				images_center.append(cv2.cvtColor(cv2.imread(img_path_center), cv2.COLOR_BGR2RGB))
				angles_center.append(float(obs[3]))
				img_path_left = data_path + obs[1].strip()
				images_left.append(cv2.cvtColor(cv2.imread(img_path_left), cv2.COLOR_BGR2RGB))
				angles_left.append(float(obs[3]) + steering_adjustment)
				img_path_right = data_path + obs[2].strip()
				images_right.append(cv2.cvtColor(cv2.imread(img_path_right), cv2.COLOR_BGR2RGB))
				angles_right.append(float(obs[3]) - steering_adjustment)

			images = np.array(images_center + images_left + images_right)
			angles = np.array(angles_center + angles_left + angles_right)

			yield shuffle(images, angles)


# Load data (using Udacity's provided dataset)
lines = []
with open(data_path + 'driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader, None)
	for line in reader:
		lines.append(line)

lines = np.array(lines)

# Split into training and validation data sets
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))
	
# Steer angle histogram
steer_angles = train_samples[:,3].astype(float)
plt.hist(steer_angles, bins=20, rwidth=0.8)
plt.show()

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)	

# Keras CNN
model = Sequential()

# Normalization
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))

# Cropping 60 pixels from the top and 20 from the bottom
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))

# Convolutional layers
model.add(Conv2D(24, (5, 5), strides=(2,2), activation='elu'))
model.add(Conv2D(36, (5, 5), strides=(2,2), activation='elu'))
model.add(Conv2D(48, (5, 5), strides=(2,2), activation='elu'))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Conv2D(64, (3, 3), activation='elu'))

# Dropout
model.add(Dropout(0.5))

# Flatten and FC layers
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1, activation='elu'))

# Fit generator
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size, \
					validation_data=validation_generator, validation_steps=len(validation_samples)/batch_size, \
					epochs=epochs, verbose=1)

model.save('model.h5')