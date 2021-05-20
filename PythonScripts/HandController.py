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

sys.stderr = open("C:\\Git\\Virtual-Hand\\PythonScripts\\PythonClientError_HandController.txt", "w")

# Constants
ZEROING_TIME_MS = 3000
NUM_SENSORS = 5
NUM_FINGERS = 5
NUM_LIMBS_PER_FINGER = 3
NUM_LIMBS = NUM_FINGERS * NUM_LIMBS_PER_FINGER
NUM_FEATURES = NUM_LIMBS * 2 + NUM_SENSORS

connection_handler = ClientConnectionHandler()
sensor_data = SensorReadingsListener()
sensor_data.start_thread()

models_base_name = connection_handler.input()  # "RealData15_man.model"  # TODO, make this depend on file later
FRAMES_PER_SECOND = 50  # TODO, CHANGE THIS TO SOMETHING MORE SUSTAINABLE LATER

HARD_INPUT_DOMAIN = [-1000000, 1000000]
HARD_OUTPUT_DOMAIN = [-1000000, 1000000]

connection_handler.print(
    "Waiting to zero the sensors... (" + str(round(ZEROING_TIME_MS / 1000.0, 1)) + " second delay minimum)")

connection_handler.print("Loading in the models...")
# Loads in the models while we wait for the zeroing to happen
milliseconds = int(time.time() * 1000)

# Adds the models
models = []
for finger_index in range(0, NUM_FINGERS):
    models.append([])
    for limb_index in range(0, NUM_LIMBS_PER_FINGER):
        models[finger_index].append(tf.keras.models.load_model(
            "C:\\Git Data\\Virtual-Hand-Data\\models\\" + models_base_name + "_"
            + str(finger_index) + str(limb_index) + ".model",
            custom_objects=None, compile=True, options=None
        ))
        connection_handler.print("Loaded in limb " + str(finger_index) + str(limb_index) + ".")
        connection_handler.input()

connection_handler.print("Loaded in the models.")

while time.time() * 1000 - milliseconds < ZEROING_TIME_MS:
    time.sleep(0.005)
    print()


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

connection_handler.print("Zeroed the sensors.")


# TODO, multithread the prediction of model data?

def enforce_domain(domain_list, value):
    return max(domain_list[0], min(domain_list[1], value))


running = True
while running:
    command = connection_handler.input()

    if command == "quit":
        running = False
    else:
        # Obtains limb data from the C# Unity script
        string_limb_data = connection_handler.input().split(" ")
        limb_data = []
        for i in range(0, len(string_limb_data)):
            limb_data.append(
                enforce_domain(HARD_INPUT_DOMAIN, float(string_limb_data[i]))
            )
            if i % 2 == 1:
                limb_data[i] = limb_data[i] * FRAMES_PER_SECOND

        # Obtains the sensors data
        current_sensor_data = sensor_data.get_readings_frame()  # Retrieves the sensors dictionary
        keys = sensor_data.get_key_list()
        sensors_data = []
        for k in keys:
            assert len(current_sensor_data) == len(zeros)
            sensors_data.append(
                enforce_domain(HARD_INPUT_DOMAIN, current_sensor_data[k] - zeros[k])
            )

        # Creates the features list
        features = np.array(limb_data + sensors_data)

        # Computes the velocities that the virtual hand limbs should acquire
        next_velocities = []
        for finger_index in range(0, NUM_FINGERS):
            for limb_index in range(0, NUM_LIMBS_PER_FINGER):
                to_predict = features.reshape(1, NUM_FEATURES)
                next_velocities.append(
                    enforce_domain(HARD_OUTPUT_DOMAIN,
                                   models[finger_index][limb_index].predict(to_predict)[0][0] * FRAMES_PER_SECOND)
                )

        # Prepared the velocities to send to the unity script
        string_velocities = ""
        for i in range(0, NUM_LIMBS):
            string_velocities += str(next_velocities[i]) + " "
        string_velocities = string_velocities.rstrip(" ")

        # Sends the torques to the unity script
        connection_handler.print(string_velocities)

print("Quitting...")
