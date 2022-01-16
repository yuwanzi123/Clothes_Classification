import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt

file_list = []
class_list = []

DATADIR = "/home/zz/Closet/database/data/"

# All the categories we want to detect in the Closet
CATEGORIES = ["c0_others", "c1_hat", "c2_pants", "c3_shoes", "c4_skirt",
	      "c5_tshirt"]

# The size of the images that our neural network will use
IMG_SIZE = 50

# Checking for all images in the data folder
for category in CATEGORIES :
	path = os.path.join(DATADIR, category)
	for img in os.listdir(path):
		img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

training_data = []

def create_training_data():
	for category in CATEGORIES :
		path = os.path.join(DATADIR, category)
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			try :
				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
				new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
				training_data.append([new_array, class_num])
			except Exception as e:
				pass
create_training_data()

print("The size of training data is: " + str(len(training_data)))

random.shuffle(training_data)

X = [] #features
y = [] #labels
y = np.array(y)

for features, label in training_data:
	X.append(features)
	#y.append(label)
	y = np.append(y, label)
	#np.array((y, label))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print("The shape of X is: ")
print(X.shape)
print("The shape of y is: ")
print(y.shape)

'''
# Creating the files containing all the information about your model
pickle_out = open("X.pickle", "wb")     #创建pickle，write bytes
pickle.dump(X, pickle_out)              #装填，在这里把X和以X为基础的pickle_out也装进去了
pickle_out.close()                      #密封

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")      #read bytes
X = pickle.load(pickle_in)


# Opening the files about data
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))
'''

# normalizing data (a pixel goes from 0 to 255)
X = X/255.0

# Building the model
model = Sequential()

# 3 convolutional layers
model.add(Conv2D(32, (3, 3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 2 hidden layers
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))


# The output layer with 6 neurons, for 6 classes
model.add(Dense(6))
model.add(Activation("softmax"))


# Compiling the model using some basic parameters
model.compile(loss="sparse_categorical_crossentropy",
				optimizer="adam",
				metrics=["accuracy"])

# Training the model, with 40 iterations
# validation_split corresponds to the percentage of images used for the validation phase compared to all the images
history = model.fit(X, y, batch_size=32, epochs=20, validation_split=0.1)


# Saving the model
model_json = model.to_json()
with open("model.json", "w") as json_file :
	json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")

model.save('CNN.model')

# Printing a graph showing the accuracy changes during the training phase

print(history.history.keys())
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')








