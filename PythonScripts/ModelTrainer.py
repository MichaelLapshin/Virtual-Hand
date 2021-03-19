"""
[ModelTrainer.py]
@description: Script for that is called by the C# script to train the model that will control the hand.
@author: Michael Lapshin
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # To decrease amount of warning (temporary)
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy
import sys
import h5py
import math

# Reads the arguments
dataset_name, model_name = sys.argv[1:]

# Obtains data
data_set = h5py.File("./training_datasets/" + dataset_name + ".hdf5", 'r')
number_of_sensors = data_set.get("sensor").len()
number_of_limbs = data_set.get("angle").len() * data_set.get("angle")[0]

# Creates model
inputs = keras.Input(shape=(number_of_limbs + number_of_sensors, "input_data"))
hidden = layers.Dense(number_of_sensors + number_of_limbs, activation="relu")(inputs)
hidden = layers.Dense(number_of_sensors + number_of_limbs, activation="relu")(hidden)
hidden = layers.Dense(number_of_sensors + number_of_limbs, activation="relu")(hidden)
outputs = layers.Dense(number_of_limbs, activation="relu")(hidden)
model = keras.Model(inputs=inputs, outputs=outputs)

current_frame_number = 0
total_number_of_frames = data_set.get("time").len()

print("Ready")  # Tells unity to start the training sequence
string_starting_angles = ""
for finger_index in range(0, 5):
    for limb_index in range(0, 3):
        string_starting_angles += " " + data_set.get("angle")[finger_index][limb_index][0]
print(string_starting_angles[1:])

input_train = data_set.get("sensor")
input_train_frametime = data_set.get("time")
input_test = data_set.get("angle")

# Constants
# RANGE_FAILED_FRAMES = 20  # todo, implement this later
# MAX_FAILED_FRAMES = 5
ANGLE_THRESHOLD = 10 / 180.0 * math.pi  # +- 10 degrees from actual angle

while current_frame_number != total_number_of_frames:
    differences = []
    current_frame_number = 0
    failed_frame = False

    # Gathers data from Unity as it runs the input without the model
    while not failed_frame:
        print(data_set.get("time")[current_frame_number])

        # Obtains data from the C# Unity script
        string_limb_data = input().rstrip(" \n\r").split(" ")
        limb_data = []
        for i in range(0, len(string_limb_data)):
            limb_data.append(float(string_limb_data[i]))

        # Obtains the expacted output
        expected_output_case = []
        for finger_index in range(0, 5):
            for limb_index in range(0, 3):
                expected_output_case.append(data_set.get("angle")[finger_index][limb_index][current_frame_number])

        # Runs the data through the model
        differences.append(data_set.get("angle")[]; model.predict())  # todo, work on the model feeding

        # todo
        # - While the frames are not failing, go collect their differences from the actual angles
        # - When the current frame fails, use the differences to compute the average loss; correct the model ("step down the gradient")
        # -
        pass

    # Updates the model here
