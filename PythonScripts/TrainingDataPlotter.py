"""
[TrainingDataPlotter.py]
@description: Class solely for displaying and saving the training data.
@author: Michael Lapshin
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PlotData:

    def __init__(self, training_name, show_images=True, save_as_images=False):
        reader = h5py.File("C:\\Git Data\\Virtual-Hand-Data\\training_datasets\\" + training_name + ".hdf5", 'r')

        # Plots the data below
        plt.title("Time")
        plt.plot(np.array(reader.get("time")))
        plt.xlabel("Frame")
        plt.ylabel("Milliseconds since Start")
        if save_as_images:
            plt.savefig(training_name + "_Time.png", bbox_inches='tight')
        if show_images:
            plt.show()

        finger_name = ["Thumb Finger", "Index Finger", "Middle Finger", "Ring Finger", "Pinky Finger"]
        limb_part = ["proximal", "middle", "distal"]
        finger_label = ["thumb", "index", "middle", "ring", "pinky"]

        # Saves/displays the graphs
        for sensor in range(0, len(list(reader.get("sensor")))):
            plt.plot(np.array(reader.get("sensor")[sensor]))
        plt.legend(finger_label)
        plt.title(label=training_name + " Sensors")
        plt.xlabel("Frame")
        plt.ylabel("Sensor Reading")
        if save_as_images:
            plt.savefig(training_name + "_Sensors.png", bbox_inches='tight')
        if show_images:
            plt.show()

        for finger in range(0, len(list(reader.get("angle")))):
            plt.plot(np.array(reader.get("angle")[finger][0]))
            plt.plot(np.array(reader.get("angle")[finger][1]))
            plt.plot(np.array(reader.get("angle")[finger][2]))
            plt.title(label=training_name + " " + finger_name[finger])
            plt.xlabel("Frame")
            plt.ylabel("Angle (radians)")
            plt.legend(limb_part)
            if save_as_images:
                plt.savefig(training_name + "_" + finger_name[finger] + "_angle.png", bbox_inches='tight')
            if show_images:
                plt.show()

        for finger in range(0, len(list(reader.get("velocity")))):
            plt.plot(np.array(reader.get("velocity")[finger][0]))
            plt.plot(np.array(reader.get("velocity")[finger][1]))
            plt.plot(np.array(reader.get("velocity")[finger][2]))
            plt.title(label=training_name + " " + finger_name[finger])
            plt.xlabel("Frame")
            plt.ylabel("Velocity (radians)")
            plt.legend(limb_part)
            if save_as_images:
                plt.savefig(training_name + "_" + finger_name[finger] + "_velocity.png", bbox_inches='tight')
            if show_images:
                plt.show()

        for finger in range(0, len(list(reader.get("acceleration")))):
            plt.plot(np.array(reader.get("acceleration")[finger][0]))
            plt.plot(np.array(reader.get("acceleration")[finger][1]))
            plt.plot(np.array(reader.get("acceleration")[finger][2]))
            plt.title(label=training_name + " " + finger_name[finger])
            plt.xlabel("Frame")
            plt.ylabel("Acceleration (radians)")
            plt.legend(limb_part)
            if save_as_images:
                plt.savefig(training_name + "_" + finger_name[finger] + "_acceleration.png", bbox_inches='tight')
            if show_images:
                plt.show()

        reader.close()
