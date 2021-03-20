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
import time

# print(sys.path)
# Reads the arguments
# print(sys.argv)
# dataset_name = sys.argv[1]
# model_name = sys.argv[2]
dataset_name = input()
model_name = input()

# Obtains data
# data_set = h5py.File("./training_datasets/" + dataset_name + ".hdf5", 'r')
data_set = h5py.File("C:\\Git\\Virtual-Hand\\PythonScripts\\training_datasets\\" + dataset_name + ".hdf5", 'r')
number_of_sensors = len(data_set.get("sensor"))
number_of_limbs = len(data_set.get("angle")) * len(data_set.get("angle")[0])


# Creates model
class CustomModel(keras.Sequential):
    def __init__(self):
        super(CustomModel, self).__init__()

    def test_step(self, data):
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided.
        data = self.data_adapter.expand_1d(data)
        # x, y, sample_weight = self.data_adapter.unpack_x_y_sample_weight(data)
        y, y_pred, sample_weight = self.data_adapter.unpack_x_y_sample_weight(data)

        with self.backprop.GradientTape() as tape:
            # y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    # def make_test_function(self):
    #     return super.make_test_function()

    # def fit(self,
    #         x=None,
    #         y=None,
    #         batch_size=None,
    #         epochs=1,
    #         verbose=1,
    #         callbacks=None,
    #         validation_split=0.,
    #         validation_data=None,
    #         shuffle=True,
    #         class_weight=None,
    #         sample_weight=None,
    #         initial_epoch=0,
    #         steps_per_epoch=None,
    #         validation_steps=None,
    #         validation_batch_size=None,
    #         validation_freq=1,
    #         max_queue_size=10,
    #         workers=1,
    #         use_multiprocessing=False):
    #     return super.fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data,
    #                      shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps,
    #                      validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)


# Generate model
model = CustomModel()
# model = keras.Sequential()

# print("limbs", number_of_limbs)
# print("sensors", number_of_sensors)
model.add(keras.Input(shape=(number_of_limbs * 2 + number_of_sensors,)))
model.add(layers.Dense(number_of_sensors + number_of_limbs, activation=keras.activations.linear,
                       bias_initializer=keras.initializers.zeros, kernel_initializer=keras.initializers.zeros,
                       dtype=tensorflow.float32, trainable=True))
model.add(layers.Dense(number_of_sensors + number_of_limbs, activation=keras.activations.linear,
                       bias_initializer=keras.initializers.zeros, kernel_initializer=keras.initializers.zeros,
                       dtype=tensorflow.float32, trainable=True))
model.add(layers.Dense(number_of_sensors + number_of_limbs, activation=keras.activations.linear,
                       bias_initializer=keras.initializers.zeros, kernel_initializer=keras.initializers.zeros,
                       dtype=tensorflow.float32, trainable=True))
model.add(layers.Dense(number_of_limbs, ))  # output layer

unity_frame_angles = []
expected_frame_angles = []


def keras_custom_loss_function(y_actual, y_predicted):
    loss_result = tensorflow.keras.losses.mean_squared_error(expected_frame_angles, unity_frame_angles)
    return loss_result


# model.compile(loss=keras_custom_loss_function, metrics=["accuracy"]) #todo, fix?
model.compile()

time.sleep(0.5)  # todo, to remove?

print("Ready")  # Tells unity to start the training sequence
string_starting_angles = ""
for finger_index in range(0, 5):
    for limb_index in range(0, 3):
        string_starting_angles += " " + str(data_set.get("angle")[finger_index][limb_index][0])
string_starting_angles = string_starting_angles.lstrip(" ")
print(string_starting_angles)

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
        limb_data_angles = limb_data[::2]  # todo, set step to 3 when you introduce acceleration

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

        # inn = [[0 for i in range(0, number_of_limbs * 2 + number_of_sensors)]]
        # ott = [[0 for i in range(0, number_of_limbs * 2 + number_of_sensors)]]

        # Computes the loss of the current sequence
        # loss_result = keras.losses.mean_squared_error(expected_frame_angles, unity_frame_angles)
        # loss_result = model.compiled_loss(inn, ott)

        # Computes the gradient based on the loss_result and the current model weights
        # grads = tensorflow.GradientTape.gradient(loss_result, model.trainable_weights)

        # Updates the model weights using the gradient
        # model = keras.optimizers.apply_gradients(zip(grads, model.trainable_weights), model)

        model.fit(unity_frame_time, expected_frame_angles)

        # print(model.input_shape, model.output_shape)

        # # model.fit(inn, ott, shuffle=False)
        # model.fit(x=inn, y=ott, shuffle=False)
        # TODO, change to *3 when acceleration gets introduced

        # model.train_on_batch([0 for i in range(0, number_of_limbs * 2 + number_of_sensors)], [0 for i in range(0, number_of_limbs)])

        # model.train_on_batch(grads)

        # model.compile(optimizer=keras_cus)
        # model.fit(expected_frame_angles, unity_frame_angles)

        model.save("/models/" + model_name)
