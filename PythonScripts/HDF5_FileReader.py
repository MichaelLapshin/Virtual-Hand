"""
[HDF5_FileReader.py]
@description: Script for peeking into what the HDF5 files contains.
@author: Michael Lapshin
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SAVE_AS_IMAGES = True
training_name = input("Enter the training name: ")

# General Information
reader = h5py.File("./training_datasets/" + training_name + ".hdf5", 'r')
print("Number of frames =", len(list(reader.get("time"))))
print("Keys:", reader.keys())
print("Features length:", len(reader))

# Peeks into the data
print("\nDataset Shape:")
for key in reader.keys():
    print(reader.get(key).shape)

print("\nFirst 20 timestamps of the dataset.")
for i in range(0, 20):
    print(reader.get("time")[i] / 1000)

print("\nFirst 20 sensor[0] of the dataset.")
for i in range(0, 20):
    print(reader.get("sensor")[0][i])

print("\nFirst 20 angle[0][0] of the dataset.")
for i in range(0, 20):
    print(reader.get("angle")[0][0][i])

# Plots the data below
plt.title("Time")
plt.plot(np.array(reader.get("time")))
plt.xlabel("Frame")
plt.ylabel("Milliseconds since Start")
if SAVE_AS_IMAGES:
    plt.savefig(training_name+"_Time.png", bbox_inches='tight')
plt.show()

finger_name = ["Thumb Finger", "Index Finger", "Middle Finger", "Ring Finger", "Pinky Finger"]
limb_part = ["proximal", "middle", "distal"]
finger_label = ["thumb", "index", "middle", "ring", "pinky"]

# Saves/displays the graphs
for sensor in range(0, 5):
    plt.plot(np.array(reader.get("sensor")[sensor]))
plt.legend(finger_label)
plt.title(label=training_name + " Sensors")
plt.xlabel("Frame")
plt.ylabel("Sensor Reading")
if SAVE_AS_IMAGES:
    plt.savefig(training_name+"_Sensors.png", bbox_inches='tight')
plt.show()

for finger in range(0, 5):
    plt.plot(np.array(reader.get("angle")[finger][0]))
    plt.plot(np.array(reader.get("angle")[finger][1]))
    plt.plot(np.array(reader.get("angle")[finger][2]))
    plt.title(label=training_name + " " + finger_name[finger])
    plt.xlabel("Frame")
    plt.ylabel("Angle (radians)")
    plt.legend(limb_part)
    if SAVE_AS_IMAGES:
        plt.savefig(training_name + "_"+finger_name[finger]+"_angle.png", bbox_inches='tight')
    plt.show()

for finger in range(0, 5):
    plt.plot(np.array(reader.get("velocity")[finger][0]))
    plt.plot(np.array(reader.get("velocity")[finger][1]))
    plt.plot(np.array(reader.get("velocity")[finger][2]))
    plt.title(label=training_name + " " + finger_name[finger])
    plt.xlabel("Frame")
    plt.ylabel("Velocity (radians)")
    plt.legend(limb_part)
    if SAVE_AS_IMAGES:
        plt.savefig(training_name + "_"+finger_name[finger]+"_velocity.png", bbox_inches='tight')
    plt.show()

for finger in range(0, 5):
    plt.plot(np.array(reader.get("acceleration")[finger][0]))
    plt.plot(np.array(reader.get("acceleration")[finger][1]))
    plt.plot(np.array(reader.get("acceleration")[finger][2]))
    plt.title(label=training_name + " " + finger_name[finger])
    plt.xlabel("Frame")
    plt.ylabel("Acceleration (radians)")
    plt.legend(limb_part)
    if SAVE_AS_IMAGES:
        plt.savefig(training_name + "_"+finger_name[finger]+"_acceleration.png", bbox_inches='tight')
    plt.show()

reader.close()
