"""
[HDF5_FileReader.py]
@description: Script for peeking into what the HDF5 files contains.
@author: Michael Lapshin
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np

training_name = input("Enter the training name: ")

# General Information
reader = h5py.File("./training_datasets/" + training_name + ".hdf5", 'r')
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

print("\nFirst 20 angle[1][1] of the dataset.")
for i in range(0, 20):
    print(reader.get("angle")[1][1][i])

# Plots the data below
plt.title = "Time"
plt.plot(np.array(reader.get("time")))
plt.show()

for sensor in range(0, 5):
    plt.title = "Sensors: "
    plt.plot(np.array(reader.get("sensor")[sensor]))
plt.show()

for finger in range(0, 5):
    plt.plot(np.array(reader.get("angle")[finger][0]))
    plt.plot(np.array(reader.get("angle")[finger][1]))
    plt.plot(np.array(reader.get("angle")[finger][2]))
    plt.title = "Finger: " + str(finger)
    plt.show()

reader.close()
