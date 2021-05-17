"""
[HDF5_FileReader.py]
@description: Script for peeking into what the HDF5 files contains.
@author: Michael Lapshin
"""

import h5py
from TrainingDataPlotter import PlotData

training_name = input("Enter the training name: ")
inp = input("Save the plots? ")
save_as_images = (inp == "yes" or inp == "y" or inp == "1")

# General Information
reader = h5py.File("C:\\Git Data\\Virtual-Hand-Data\\training_datasets\\" + training_name + ".hdf5", 'r')
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

reader.close()

# Plots the data
PlotData(training_name=training_name, show_images=True, save_as_images=save_as_images)
