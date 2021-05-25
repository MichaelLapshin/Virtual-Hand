"""
[ContainedModelTrainer.py]
@description: Script for generating a model without interacting with unity.
@author: Michael Lapshin
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # To decrease amount of warnings (temporary)

import time
import threading

import numpy as np
import tensorflow as tf
import h5py
import math

from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt

# Leading in the data file
models_base_name = input("Name of the model base name (to view/create): ")
dataset_names = input("Name of the datasets (separated by spaces): ")  # "RealData15_manual2"
# dataset_names = dataset_names.split(" ")
dataset_names = ["T180_s", "T300_1_s", "T300_2_s", "T300_3_s"]
# dataset_names = ["T180_s"]
inp = input("Train a model? ")
train_model = (inp == "yes" or inp == "y" or inp == "1")

train_new_model = False
if train_model:
    inp = input("Train a new model? ")
    train_new_model = (inp == "yes" or inp == "y" or inp == "1")

default_training = True
if train_new_model:
    inp = input("Default training parameters? ")
    if not (inp == "yes" or inp == "y" or inp == "1"):
        learning_rate = float(input("Learning rate (float, e.g. 0.0005): "))
        epochs = int(input("Number of epochs (int, e.g. 5000): "))
        batch_size = int(input("Batch size (int, e.g. 32): "))
        default_training = False

# Finger information
NUM_FRAMES = None
NUM_SENSORS = 5
NUM_FINGERS = 5
NUM_LIMBS_PER_FINGER = 3
NUM_LIMBS = NUM_FINGERS * NUM_LIMBS_PER_FINGER
NUM_FEATURES = NUM_LIMBS * 2 + NUM_SENSORS

CHECKON_TIME = 60

FRAMES_DIF_COMPARE = 0
NUM_HIDDEN_NEURONS = 24
# HIDDEN_LAYERS = ["relu" for i in range(0, 3)]
# HIDDEN_LAYERS = [tf.keras.layers.LeakyReLU() for i in range(0, 3)]
# HIDDEN_LAYERS = ["linear" for i in range(0, 3)]
HIDDEN_LAYERS = ["selu" for i in range(0, 3)]


# Basic rotational velocity calculation
def rads_per_second(angle_diff, frame_rate):
    return angle_diff * frame_rate


# Shifts the data within a list. shift > 0 moves the data up the positions
def shift_data(old_list, shift=0):
    new_list = old_list[-shift::]
    new_list += old_list[:-shift:]
    return new_list


def get_training_data(dataset_name):
    data_set = h5py.File("C:\\Git Data\\Virtual-Hand-Data\\training_datasets\\" + dataset_name + ".hdf5", 'r')
    assert len(data_set["velocity"]) > 0 and data_set["velocity"] is not None

    # Finger information
    current_num_frames = len(data_set.get("time"))

    # Gathers the data (and shifts the data appropriately)
    all_features = []
    current_label_data = []
    for finger_index in range(0, NUM_FINGERS):
        current_label_data.append([])
        for limb_index in range(0, NUM_LIMBS_PER_FINGER):
            all_features.append(data_set.get("angle")[finger_index][limb_index])
            all_features.append(data_set.get("velocity")[finger_index][limb_index])

            current_label_data[finger_index].append(
                shift_data(list(data_set.get("velocity")[finger_index][limb_index]), -FRAMES_DIF_COMPARE))

    for sensor_index in range(0, NUM_SENSORS):
        all_features.append(shift_data(list(data_set.get("sensor")[sensor_index]), -FRAMES_DIF_COMPARE))

    # Closes the dataset
    data_set.close()

    # Makes sure that data dimensions are valid
    for i in range(1, len(all_features)):
        assert len(all_features[i]) == len(all_features[i - 1])

    # Creating the training data
    current_training_data = []  # Every index represents a new training feature
    for frame in range(0, current_num_frames):
        frame_data = []

        for i in range(0, len(all_features)):
            frame_data.append(all_features[i][frame])

        current_training_data.append(np.array(frame_data))

    # Crops the data to avoid wrapping data from the data shift
    if FRAMES_DIF_COMPARE != 0:
        for finger_index in range(0, NUM_FINGERS):
            for limb_index in range(0, NUM_LIMBS_PER_FINGER):
                current_label_data[finger_index][limb_index] = current_label_data[finger_index][limb_index][
                                                               1:-FRAMES_DIF_COMPARE:]
        current_training_data = current_training_data[1:-FRAMES_DIF_COMPARE:]

    print("\n===", dataset_name, "===")
    print("len(current_training_data[0]) =", len(current_training_data[0]))
    print("len(current_label_data[0][0]) == len(current_training_data) == ", len(current_label_data[0][0]))
    assert len(current_label_data[0][0]) == len(current_training_data)

    return current_training_data, current_label_data


# Gathers all of the data from all of the data sets
training_data = None
label_data = None

for dataset in dataset_names:
    c_training, c_labels = get_training_data(dataset)

    if training_data is None or len(training_data) == 0:
        training_data = c_training
    else:
        training_data += c_training

    if label_data is None or len(label_data) == 0:
        label_data = c_labels
    else:
        for finger_index in range(0, NUM_FINGERS):
            for limb_index in range(0, NUM_LIMBS_PER_FINGER):
                label_data[finger_index][limb_index] += c_labels[finger_index][limb_index]

# Finger information (more)
NUM_FRAMES = len(training_data)
assert NUM_FRAMES is not None

# Training
if default_training:
    epochs = int(3000 + 250 * math.sqrt(NUM_FRAMES))
    learning_rate = 0.0015
    batch_size = int(NUM_FRAMES * 0.15)

print("\n===== Official Training Information =====")
print("learning rate:", learning_rate)
print("batch_size:", batch_size)
print("epochs:", epochs)
print("=========================================\n")

print("\nLet the training begin!\n")

print(np.shape(label_data))
print(np.shape(training_data))

label_data = np.array(label_data)
training_data = np.array(training_data)

normalizer_layer = preprocessing.Normalization()
normalizer_layer.adapt(training_data)


def plot_loss(history, name):
    plt.plot(history.history['loss'], label='mean_absolute_error')
    plt.ylim([0, 0.01])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.title("Absolute Error for " + name)
    plt.legend()
    plt.grid(True)
    plt.show()


np.set_printoptions(precision=10, suppress=True)

models = []
# Loads a model by specification
if not train_new_model:
    for finger_index in range(0, NUM_FINGERS):
        models.append([])
        for limb_index in range(0, NUM_LIMBS_PER_FINGER):
            models[finger_index].append(
                tf.keras.models.load_model(
                    "C:\\Git Data\\Virtual-Hand-Data\\models\\" + models_base_name + "_"
                    + str(finger_index) + str(limb_index) + ".model",
                    custom_objects=None, compile=True, options=None
                )
            )


# Class used to train model on separate thread
class ModelTrainer(threading.Thread):
    def __init__(self, model, finger_index, limb_index):
        threading.Thread.__init__(self)  # calls constructor of the Thread class
        self.done = False
        self.model = model
        self.finger_index = finger_index
        self.limb_index = limb_index
        self.start()

    def run(self):
        self.history = self.model.fit(
            training_data, label_data[self.finger_index][self.limb_index],
            verbose=0,
            batch_size=batch_size,
            epochs=epochs, shuffle=True)

        # Compile the model # TODO, refine the model witha lower learning rate
        # self.model.compile(loss="mse",  # loss='mean_absolute_error',
        #               optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        #               metrics=['mean_absolute_percentage_error'])

        self.done = True


# Trains the model
threaded_models = []
if train_model:
    if train_new_model:
        for finger_index in range(0, NUM_FINGERS):
            threaded_models.append([])
            for limb_index in range(0, NUM_LIMBS_PER_FINGER):
                # Build the model
                model_layers = [layers.Input(shape=(NUM_FEATURES,))]
                for act in HIDDEN_LAYERS:
                    model_layers.append(layers.Dense(NUM_HIDDEN_NEURONS, activation=act, bias_initializer='zeros'))
                model_layers.append(layers.Dense(1))

                model = keras.Sequential(model_layers)
                model.summary()

                # Compile the model
                model.compile(loss="mse",  # loss='mean_absolute_error',
                              optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                              metrics=['mean_absolute_percentage_error'])

                threaded_models[finger_index].append(
                    ModelTrainer(model=model, finger_index=finger_index, limb_index=limb_index))

        done_count = 0
        while done_count != NUM_LIMBS:
            time.sleep(CHECKON_TIME)

            done_count = 0
            for finger_index in range(0, NUM_FINGERS):
                for limb_index in range(0, NUM_LIMBS_PER_FINGER):
                    if threaded_models[finger_index][limb_index].done:
                        done_count += 1

            print(done_count, " are done processing.")

        for finger_index in range(0, NUM_FINGERS):
            models.append([])
            for limb_index in range(0, NUM_LIMBS_PER_FINGER):
                history = threaded_models[finger_index][limb_index].history
                print("-----------------------------------------------------------------------------------------------")
                print("Done with", str(finger_index), str(limb_index))

                test_absolute_error, test_percent_error = threaded_models[finger_index][limb_index].model.evaluate(
                    x=training_data, y=label_data[finger_index][limb_index], verbose=1)

                print("Test Absolute Error:", test_absolute_error,
                      "    Test Percent Error:", test_percent_error)

                plot_loss(history, name=str(finger_index) + str(limb_index))

                # Adds the model to the list
                models[finger_index].append(threaded_models[finger_index][limb_index].model)

    else:  # if train an existing model
        for finger_index in range(0, NUM_FINGERS):
            for limb_index in range(0, NUM_LIMBS_PER_FINGER):
                model = models[finger_index][limb_index]

                # Fits the model
                history = model.fit(
                    training_data, label_data,
                    verbose=0,
                    batch_size=batch_size,
                    epochs=epochs, shuffle=True)

                test_absolute_error, test_percent_error = model.evaluate(x=training_data, y=label_data, verbose=1)

                print("Done with", str(finger_index), str(limb_index))
                print("Test Absolute Error:", test_absolute_error,
                      "    Test Percent Error:", test_percent_error)

                plot_loss(history, name=str(finger_index) + str(limb_index))

print("\nGot the models.\n")


# Manual review
def plot_predictions(model, frames, name):
    data = []
    for i in range(0, len(frames)):
        to_predict = training_data[i].reshape(1, NUM_FEATURES)
        data.append(model.predict(to_predict)[0][0])

    plt.plot(data)
    plt.xlabel('Frame')
    plt.ylabel("Prediction")
    plt.title('Predictions: ' + name)
    plt.grid(True)
    plt.show()


review = True
while review:
    inp = input("Enter a command: ['save', 'quit'/'exit', 'predict']: ")

    if inp == "save":
        for finger_index in range(0, NUM_FINGERS):
            for limb_index in range(0, NUM_LIMBS_PER_FINGER):
                models[finger_index][limb_index].save(
                    "C:\\Git Data\\Virtual-Hand-Data\\models\\" + models_base_name + "_"
                    + str(finger_index) + str(limb_index) + ".model")
    elif inp == "quit" or inp == "exit":
        review = False
    elif inp == "predict":
        for finger_index in range(0, NUM_FINGERS):
            for limb_index in range(0, NUM_LIMBS_PER_FINGER):
                plot_predictions(models[finger_index][limb_index], training_data,
                                 name=str(finger_index) + str(limb_index))
