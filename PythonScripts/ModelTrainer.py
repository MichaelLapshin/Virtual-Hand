"""
[ModelTrainer.py]
@description: Script for that is called by the C# script to train the model that will control the hand.
@author: Michael Lapshin
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # To decrease amount of warnings (temporary)
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
# class CustomModel(keras.Sequential):
#     def __init__(self):
#         super(CustomModel, self).__init__()
#
#     def test_step(self, data):
#         data = super.data_adapter.expand_1d(data)
#         x, y, sample_weight = super.data_adapter.unpack_x_y_sample_weight(data)
#
#         y_pred = self(x, training=False)
#         # Updates stateful loss metrics.
#         self.compiled_loss(
#             y, y_pred, sample_weight, regularization_losses=self.losses)
#
#         self.compiled_metrics.update_state(y, y_pred, sample_weight)
#         return {m.name: m.result() for m in self.metrics}
#
#     def make_test_function(self):
#         return super.make_test_function()


# Generate model
model = keras.Sequential()

model.add(keras.Input(shape=(number_of_limbs * 2 + number_of_sensors,)))
model.add(layers.Dense(number_of_sensors + number_of_limbs, activation=keras.activations.linear))
model.add(layers.Dense(number_of_sensors + number_of_limbs, activation=keras.activations.linear))
model.add(layers.Dense(number_of_sensors + number_of_limbs, activation=keras.activations.linear))
model.add(layers.Dense(number_of_limbs))  # output layer

model.compile(optimizer=keras.optimizers.Adam, loss=keras.losses.mean_squared_error, metrics=["accuracy"])

print("Ready")  # Tells unity to start the training sequence
string_starting_angles = ""
for finger_index in range(0, 5):
    for limb_index in range(0, 3):
        string_starting_angles += " " + data_set.get("angle")[finger_index][limb_index][0]
print(string_starting_angles[1:])

input_train_sensors = data_set.get("sensor")
input_train_frame_time = data_set.get("time")
input_train_angles = data_set.get("angle")

# Constants
# RANGE_FAILED_FRAMES = 20  # todo, implement this later
# MAX_FAILED_FRAMES = 5
ANGLE_THRESHOLD = 10 / 180.0 * math.pi  # +- 10 degrees from actual angle

current_frame_number = 0
total_number_of_frames = data_set.get("time").len()

while current_frame_number != total_number_of_frames:
    unity_frame_angles = []
    expected_frame_angles = []
    current_frame_number = 0
    failed_frame = False

    # Gathers data from Unity as it runs the input without the model
    while not failed_frame:
        is_unity_ready = input()
        if is_unity_ready != "Ready":
            print("ERROR. C# Unity Script sent improper command. 'Ready' not receives. Received " + is_unity_ready + " instead.")

        print(data_set.get("time")[current_frame_number])

        # Receives frame capture time from unity
        unity_frame_time = int(input())

        # Obtains limb data from the C# Unity script
        string_limb_data = input().rstrip(" \n\r").split(" ")
        limb_data = []
        for i in range(0, len(string_limb_data)):
            limb_data.append(float(string_limb_data[i]))

        # Obtains the expected output
        expected_output_angles = []
        for finger_index in range(0, 5):
            for limb_index in range(0, 3):
                expected_output_angles.append(data_set.get("angle")[finger_index][limb_index][current_frame_number])

        # Extracts angles from unity model for comparison
        limb_data_angles = limb_data[::3]  # todo, set step to 3 when you introduce acceleration

        # Assert that expected and limb data is of the same length
        assert len(limb_data_angles) == len(expected_output_angles)

        # Time condition
        passed_time_condition = False
        if input_train_frame_time[current_frame_number] < unity_frame_time < input_train_frame_time[current_frame_number + 1]:
            passed_time_condition = True

        # Angle condition
        passed_angle_condition = True
        for i in range(0, len(limb_data)):
            if expected_output_angles[i] - ANGLE_THRESHOLD > limb_data_angles[i] \
                    or limb_data_angles[i] > expected_output_angles[i] + ANGLE_THRESHOLD:
                passed_angle_condition = False
                break

        # Loop-3 condition
        if current_frame_number >= total_number_of_frames:
            print("Quit")
            break
        elif not passed_time_condition or not passed_angle_condition:
            print("Reset")
            failed_frame = True
            break
        else:
            print("Next")
            expected_frame_angles.append(expected_output_angles)
            unity_frame_angles.append(limb_data_angles)

            # Preparing sensors inputs
            current_sensor_readings = [input_train_sensors[i][current_frame_number] for i in range(0, number_of_sensors)]
            # for i in range(0, number_of_sensors):
            #     current_sensor_readings.append(input_train_sensors[i][current_frame_number])

            # Runs the data through the model
            current_model_inputs = numpy.array(limb_data + current_sensor_readings)
            next_torques = model.predict(current_model_inputs,
                                         batch_size=(1, len(current_model_inputs)))  # Predicts the next torques to apply

            # Prepared the torques to send to the unity script
            string_torques = ""
            for i in range(0, number_of_limbs):
                string_torques += next_torques[i] + " "
            string_torques = string_torques.rstrip(" ")

            # Sends the torques to the unity script
            print(string_torques)

            # todo
            # - While the frames are not failing, go collect their differences from the actual angles
            # - When the current frame fails, use the differences to compute the average loss; correct the model ("step down the gradient")

    if not failed_frame:
        model.save("/models/" + model_name + "_CompletedTraining")
        break
    else:
        """
        Data for training:
            - unity_frame_angles[frame#]: angles of the limbs within unity
            - expected_frame_angles[frame#]: and expected limb angles
        
        What must do:
            - Compute the loss and gradient using the training data from above. Might require to overwrite Model.train_on_batch()
        """

        # Computes the loss of the current sequence
        loss_result = keras.losses.mean_squared_error(expected_frame_angles, unity_frame_angles)

        # Computes the gradient based on the loss_result and the current model weights
        grads = tensorflow.GradientTape.gradient(loss_result, model.trainable_weights)

        # Updates the model weights using the gradient
        keras.optimizers.Adam.apply_gradients(zip(grads, model.trainable_weights))

        model.save("/models/" + model_name)
