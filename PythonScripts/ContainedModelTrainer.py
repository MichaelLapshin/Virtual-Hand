"""
[ContainedModelTrainer.py]
@description: Script for generating a model without interacting with unity.
@author: Michael Lapshin
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # To decrease amount of warnings (temporary)

import numpy as np
import tensorflow as tf
import h5py
import math
import time
import sys

from tensorflow.keras import layers
from tensorflow import keras

# Leading in the data file
dataset_name = "RealData15_smoothed"  # input("Name of the dataset: ")
data_set = h5py.File("C:\\Git\\Virtual-Hand\\PythonScripts\\training_datasets\\" + dataset_name + ".hdf5", 'r')
assert len(data_set["velocity"]) > 0 and data_set["velocity"] is not None
DATA_FRAMES_PER_SECOND = 50

# Finger information
NUM_FRAMES = len(data_set.get("time"))
NUM_SENSORS = len(data_set.get("sensor"))
NUM_FINGERS = len(data_set.get("angle"))
NUM_LIMBS_PER_FINGER = len(data_set.get("angle")[0])
NUM_LIMBS = NUM_FINGERS * NUM_LIMBS_PER_FINGER
NUM_FEATURES = NUM_LIMBS * 2 + NUM_SENSORS

FRAMES_DIF_COMPARE = 5
NUM_HIDDEN_NEURONS = 128
HIDDEN_LAYERS = ["selu" for i in range(0, 10)]


def rads_per_second(angle_diff, frame_rate):
    return angle_diff * frame_rate


# Leading the training data
training_data = []  # Every index represents a new training feature
for frame in range(0, NUM_FRAMES):
    frame_data = []

    for finger_index in range(0, NUM_FINGERS):  # TODO, look over this
        for limb_index in range(0, NUM_LIMBS_PER_FINGER):
            frame_data.append(data_set.get("angle")[finger_index][limb_index][frame])
            frame_data.append(
                rads_per_second(data_set.get("velocity")[finger_index][limb_index][frame], DATA_FRAMES_PER_SECOND))

    for sensor_index in range(0, NUM_SENSORS):
        frame_data.append(list(data_set.get("sensor"))[sensor_index][frame])

    # for i in range(0, len(frame_data)):
    #     frame_data[i] = np.array([frame_data[i]])

    training_data.append(np.array(frame_data))

# for frame in range(0,NUM_FRAMES):
#     training_data[frame] = training_data[frame]

label_data = []
for i in range(1 + FRAMES_DIF_COMPARE, len(training_data)):
    label_data.append(training_data[i][:NUM_LIMBS * 2:2])
training_data = training_data[1:-FRAMES_DIF_COMPARE:]

# TODO, temp edit
# label_data = label_data[:50:]
# training_data = training_data[:50:]

print(len(label_data))
print(len(training_data))

label_data = np.array(label_data)
training_data = np.array(training_data)

# TODO, AAAAAAAAAAAAAAAAAAAAAAAAAAA
# (train_images, train_labels), (
#     test_images,
#     test_labels) = keras.datasets.fashion_mnist.load_data()  # Splits the data into the trainins and test sets accordingly
#
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
#
# # Data preprocessing
# # Explanation: make the data on a scale of 0-1 so that the weights of 0-1 within the network don't struggle as much to shift the value passed around
# train_images = train_images / 255.0
# test_images = test_images / 255.0
#
# print(type(train_images[0][0][0]))
# print(type(train_images[0][0]))
# print(type(train_images[0]))
# print(type(train_images))
#
# # Building a model
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),  # input later (1)
#     keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
#     keras.layers.Dense(10, activation='softmax')  # output layer (3)
# ])  # creates the layers of the model
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',              metrics=['accuracy'])
# model.fit(train_images, train_labels,          epochs=1)
#
# test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)  # compares the model to the test data
# print('Test accuracy:', test_acc)  # prints the accuracy of the test
# print('loss:', test_loss)  # TODO, AAAAAAAAAAAAAAAAAAAAAAAAAAA
# print("")
# # print(type(training_data[0][0][0]))
# print(type(training_data[0][0]))
# print(type(training_data[0]))
# print(type(training_data))

# TODO, AAAAAAAAAAAAAAAAAAAAAAAAAAA

# Data preprocessing
# Building a model
layers = []
layers.append(keras.layers.Flatten(input_shape=(35, 1)))  # input later (1)

for act_func in HIDDEN_LAYERS:  # hidden layer (2)
    layers.append(keras.layers.Dense(128, activation=act_func,
                                     kernel_initializer=keras.initializers.zeros,
                                     bias_initializer=keras.initializers.Zeros()))

layers.append(keras.layers.Dense(15, activation='linear',
                                 kernel_initializer=keras.initializers.zeros,
                                 bias_initializer=keras.initializers.Zeros()))  # output layer (3))
model1 = keras.Sequential(layers)  # creates the layers of the model

# Compiles and trains the model
model1.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
model1.fit(training_data, label_data, batch_size=NUM_FRAMES, epochs=1000)

test_loss, test_acc = model1.evaluate(training_data, label_data, verbose=1)  # compares the model to the test data
print('Test accuracy:', test_acc)  # prints the accuracy of the test
print('loss:', test_loss)

# Manual review
review = True
while review:
    inp = input("Enter input frame to predict: ")

    if inp == "save":
        model1.save("/models/LatestModel")
    elif inp == "quit" or inp == "exit":
        review = False
    else:
        print("Input:", str(training_data[int(inp)]))
        print("Input type:", type(training_data[int(inp)]))
        print("Value type:", type(training_data[int(inp)][0]))
        print("Result:", str(model1.predict(training_data[int(inp)])))
