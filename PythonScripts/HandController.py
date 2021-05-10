"""
[HandController.py]
@description: Script obtaining the state of the virtual and and sensors, and returning an appropriate velocity array.
@author: Michael Lapshin
"""

import numpy as np
import tensorflow as tf
import h5py
import math
import time
import sys

from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers

# For connecting to the server and exchanging data with the virtual hand
from ClientConnectionHandlerV2 import ClientConnectionHandler

# For listening to the sensors
from SensorListener import SensorReadingsListener

# Constants
NUM_SENSORS = 5
NUM_FINGERS = 5
NUM_LIMBS_PER_FINGER = 3
NUM_LIMBS = NUM_FINGERS * NUM_LIMBS_PER_FINGER
NUM_FEATURES = NUM_LIMBS * 2 + NUM_SENSORS

connection_handler = ClientConnectionHandler()
sensor_data = SensorReadingsListener()
sensor_data.start_thread()

print("Waiting to zero the sensors...")
time.sleep(5)


def dict_deepcopy(dict):
    d = {}
    for k in dict.keys():
        d[k] = dict[k]
    return d


# Zeros the sensor data
zeros = None
while zeros is None:
    zeros = sensor_data.get_readings_frame()
zeros = dict_deepcopy(zeros)
sensor_data.wait4new_readings()

print("Zeroed the sensors.")

models_base_name = "RealData15_man.model"  # TODO, make this depend on file later

# Adds the models
models = []
for finger_index in range(0, NUM_FINGERS):
    for limb_index in range(0, NUM_LIMBS_PER_FINGER):
        models.append(tf.keras.models.load_model(
            "C:\\Git\\Virtual-Hand\\PythonScripts\\models\\" + models_base_name + "_"
            + str(finger_index) + str(limb_index) + ".model",
            custom_objects=None, compile=True, options=None
        ))

# TODO, multithread the prediction of model data?

running = True
while running:
    command = connection_handler.input()

    if command == "quit":
        running = False
    else:
        # Obtains limb data from the C# Unity script
        string_limb_data = connection_handler.input().rstrip(" $").split(" ")
        limb_data = []
        for i in range(0, len(string_limb_data)):
            limb_data.append(float(string_limb_data[i]))

        # Obtains the sensors data
        current_sensor_data = sensor_data.get_readings_frame()  # Retrieves the sensors dictionary
        keys = sensor_data.get_key_list()
        sensors_data = []
        for k in keys:
            sensors_data.append(current_sensor_data[k] - zeros[k])

        # Creates the features list
        features = limb_data + sensors_data

        # Computes the velocities that the virtual hand limbs should acquire
        next_velocities = []
        for finger_index in range(0, NUM_FINGERS):
            for limb_index in range(0, NUM_LIMBS_PER_FINGER):
                to_predict = features.reshape(1, NUM_FEATURES)
                models[finger_index][limb_index].predict(to_predict)

        # Prepared the velocities to send to the unity script
        string_velocities = ""
        for i in range(0, NUM_LIMBS):
            string_velocities += str(next_velocities[i]) + " "
        string_velocities = string_velocities.rstrip(" ")

        # Sends the torques to the unity script
        connection_handler.print(string_velocities)

print("Quitting...")
