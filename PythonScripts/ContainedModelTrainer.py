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

from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt

# Leading in the data file
models_base_name = input("Name of the model base name (to view/create): ")
dataset_name = input("Name of the dataset: ")  # "RealData15_manual2"
inp = input("Train a model? ")
train_model = (inp == "yes" or inp == "y" or inp == "1")

train_new_model = False
if train_model:
    inp = input("Train a new model? ")
    train_new_model = (inp == "yes" or inp == "y" or inp == "1")

# todo, works well
epochs = 1200
learning_rate = 0.0004
refined_learning_rate = 0.00005
batch_size = 744
# epochs = 800 # Todo, keep this data
# learning_rate = 0.0005 # Todo, keep this data
# batch_size = 100 # Todo, keep this data
epochs = 1200
learning_rate = 0.0004
batch_size = 744
if train_new_model:
    inp = input("Default values for the training? ")
    if not (inp == "yes" or inp == "y" or inp == "1"):
        learning_rate = float(input("Learning rate (float, e.g. 0.0005): "))
        epochs = int(input("Number of epochs (int, e.g. 5000): "))
        batch_size = int(input("Batch size (int, e.g. 32): "))

data_set = h5py.File("C:\\Git Data\\Virtual-Hand-Data\\training_datasets\\" + dataset_name + ".hdf5", 'r')
assert len(data_set["velocity"]) > 0 and data_set["velocity"] is not None
DATA_FRAMES_PER_SECOND = 50

# Finger information
NUM_FRAMES = len(data_set.get("time"))
NUM_SENSORS = len(data_set.get("sensor"))
NUM_FINGERS = len(data_set.get("angle"))
NUM_LIMBS_PER_FINGER = len(data_set.get("angle")[0])
NUM_LIMBS = NUM_FINGERS * NUM_LIMBS_PER_FINGER
NUM_FEATURES = NUM_LIMBS * 2 + NUM_SENSORS

CHECKON_TIME = 10
FRAMES_DIF_COMPARE = 6
# NUM_HIDDEN_NEURONS = 164
# HIDDEN_LAYERS = ["relu" for i in range(0, 32)]
NUM_HIDDEN_NEURONS = 70
HIDDEN_LAYERS = ["relu" for i in range(0, 8)]

# Basic rotational velocity calculation
def rads_per_second(angle_diff, frame_rate):
    return angle_diff * frame_rate


# Shifts the data within a list. shift > 0 moves the data up the positions
def shift_data(old_list, shift=0):
    new_list = old_list[-shift::]
    new_list += old_list[:-shift:]
    return new_list


# Gathers the data (and shifts the data appropriately)
all_features = []
label_data = []
for finger_index in range(0, NUM_FINGERS):
    label_data.append([])
    for limb_index in range(0, NUM_LIMBS_PER_FINGER):
        all_features.append(data_set.get("angle")[finger_index][limb_index])
        all_features.append(data_set.get("velocity")[finger_index][limb_index])

        label_data[finger_index].append(
            shift_data(list(data_set.get("velocity")[finger_index][limb_index]), -FRAMES_DIF_COMPARE))

for sensor_index in range(0, NUM_SENSORS):
    all_features.append(shift_data(list(data_set.get("sensor")[sensor_index]), -FRAMES_DIF_COMPARE))

# Makes sure that data dimensions are valid
for i in range(1, len(all_features)):
    assert len(all_features[i]) == len(all_features[i - 1])

# Creating the training data
training_data = []  # Every index represents a new training feature
for frame in range(0, NUM_FRAMES):
    frame_data = []

    for i in range(0, len(all_features)):
        frame_data.append(all_features[i][frame])

    training_data.append(np.array(frame_data))

# training_data = training_data[:100:]  # TODO< REMOVE THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Crops the data to avoid wrapping data from the data shift
for finger_index in range(0, NUM_FINGERS):
    for limb_index in range(0, NUM_LIMBS_PER_FINGER):
        label_data[finger_index][limb_index] = label_data[finger_index][limb_index][1:-FRAMES_DIF_COMPARE:]
training_data = training_data[1:-FRAMES_DIF_COMPARE:]

print("len(training_data[0]) =", len(training_data[0]))
print("len(label_data[0][0]) == len(training_data) == ", len(label_data[0][0]))
assert len(label_data[0][0]) == len(training_data)

print("\nLet the training begin!\n")

label_data = np.array(label_data)
training_data = np.array(training_data)

normalizer_layer = preprocessing.Normalization()
normalizer_layer.adapt(training_data)


def plot_loss(history, name):
    plt.plot(history.history['loss'], label='mean_absolute_error')
    # plt.plot(history.history['val_loss'], label='val_loss')
    # plt.plot(history.history['mean_absolute_error'], label='mean_absolute_error')
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
