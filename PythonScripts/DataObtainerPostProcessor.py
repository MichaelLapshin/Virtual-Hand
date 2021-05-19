"""
[DataObtainerPostProcessor.py]
@description: Script for processing the training data (eg. smoothing the data, velocity, acceleration, etc.)
@author: Michael Lapshin
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # To remove the redundant warnings
import numpy
import h5py
import math
import numpy as np
import time

from TrainingDataPlotter import PlotData

"""
# Training data format
time | sensor | angle | velocity | acceleration

time: time in nanoseconds since the start of the training sequence
sensor: [sensor1,sensor2,...,sensorN]
    - zeroed-sensor readings
angle: [[[fingerAngles]*fingerLimb]*fingers]
    - angles of the finger limbs
velocity: [[[fingerAngularVelocities]*fingerLimb]*fingers]
    - velocity of the finger limbs
acceleration: [[[fingerAngularAcceleration]*fingerLimb]*fingers] 
    - acceleration of the finger limbs
* Note: the angle(and its derivatives) lists is in the 5 by 3 by N format
"""

# Questionaire
ORIGINAL_FILE_NAME = input("Original file name: ")
NEW_FILE_NAME = ORIGINAL_FILE_NAME.rstrip("_raw") + "_" + input("What extension should the new file name have? ")
ORIGINAL_FRAME_RATE = int(input("Original dataset framerate: "))
NEW_FRAME_RATE = int(input("New frame rate (for data interpolation): "))
SMOOTH_FRAMES = int(input("How many frames far should the smoothing look? "))  # 5 frames?
SMOOTH_DIF_THRESHOLD = float(
    input("What degree difference in the frames should the smooth? ")) * math.pi / 180.0  # 25 degrees?
loop_times = int(input("How many times should the settings be applied to the data? "))

# For visual display
inp = input("Manual control of the iterations? ")
manual_control = (inp == "yes" or inp == "y" or inp == "1")

draw_result = True
if not manual_control:
    inp = input("Draw final result? ")
    draw_result = (inp == "yes" or inp == "y" or inp == "1")

# Constants
NUM_FINGERS = 5
NUM_LIMBS_PER_FINGER = 3
NUM_SENSORS = 5
eps = np.finfo(np.float32).eps.item()


def average(data_list):
    sum = 0
    for i in data_list:
        sum += i
    return float(sum) / len(data_list)


# Computes running average of sensor data by looking running_average frames ahead
def compute_forward_running_average(original_list, running_average=1):
    assert running_average > 0

    new_list = []

    for s in range(0, len(original_list) - running_average + 1):
        new_list.append(average(original_list[s:running_average:]))

    for i in original_list[len(original_list) - running_average::]:
        new_list.append(i)

    assert len(original_list) == len(new_list)

    return new_list


# Original lists which the post-processing will be based off of
def float_int_unknownArray2list(u_list):
    if type(u_list) == float or type(u_list) == int \
            or type(u_list) == numpy.int32 or type(u_list) == numpy.float64:
        return u_list
    elif type(u_list) != list:
        u_list = list(u_list)

    for i in range(0, len(u_list)):
        u_list[i] = float_int_unknownArray2list(u_list[i])

    return u_list


def smooth_data(old_frame_rate, new_frame_rate, old_time_list, old_sensor_list, old_angle_list, times_remain=1):
    if times_remain <= 0:
        return old_time_list, old_sensor_list, old_angle_list

    # Empty lists to be filled by the program
    time_list = []
    sensor_list = [[] for a in range(0, NUM_FINGERS)]
    angle_list = [[[] for b in range(0, NUM_LIMBS_PER_FINGER)] for a in range(0, NUM_FINGERS)]

    # Smooth the angular data (averages out the angle using nearby samples)
    smooth_old_angle_list = [[[] for b in range(0, NUM_LIMBS_PER_FINGER)] for a in range(0, NUM_FINGERS)]
    for finger_index in range(0, NUM_FINGERS):
        for limb_index in range(0, NUM_LIMBS_PER_FINGER):
            for f in range(0, len(old_time_list)):
                new_angle = old_angle_list[finger_index][limb_index][f]
                frames_considered = 1

                # To the right of the subject frame
                for r in range(1, SMOOTH_FRAMES):
                    if f + r >= len(old_angle_list[finger_index][limb_index]):
                        break
                    if math.fabs(old_angle_list[finger_index][limb_index][f + r] -
                                 old_angle_list[finger_index][limb_index][f]) > SMOOTH_DIF_THRESHOLD:
                        break
                    else:
                        new_angle += old_angle_list[finger_index][limb_index][f + r] * (SMOOTH_FRAMES - r)
                        frames_considered += (SMOOTH_FRAMES - r)

                # To the left of the subject frame
                for l in range(1, SMOOTH_FRAMES):
                    if f - l < 0:
                        break
                    if math.fabs(old_angle_list[finger_index][limb_index][f - l] +
                                 old_angle_list[finger_index][limb_index][f]) > SMOOTH_DIF_THRESHOLD:
                        break
                    else:
                        new_angle += old_angle_list[finger_index][limb_index][f - l] * (SMOOTH_FRAMES - l)
                        frames_considered += (SMOOTH_FRAMES - l)

                smooth_old_angle_list[finger_index][limb_index].append(
                    new_angle / frames_considered)  # / frames_considered)

    # Adjust the frame rate (linear interpolation)
    old_ms_per_frame = 1000.0 / old_frame_rate
    new_ms_per_frame = 1000.0 / new_frame_rate

    for f in range(1, len(old_time_list)):
        # Computes difference
        time_dif = old_ms_per_frame
        current_time = f * time_dif
        previous_time = current_time - time_dif

        # Computes how many frames and the time difference between each frame to be used
        frames_in_between = float(time_dif) / new_ms_per_frame - 1  # rounds number
        time_between_frames = float(time_dif) / (frames_in_between + 1)

        # Appends time data
        for c in range(0, round(frames_in_between) + 1):
            time_list.append(int(previous_time + time_between_frames * c))

        # Appends sensor data
        for sensor_index in range(0, NUM_SENSORS):
            # Retrieves relevant sensor values
            previous_sensor = old_sensor_list[sensor_index][f - 1]
            current_sensor = old_sensor_list[sensor_index][f]

            # Computes difference
            sensor_dif = current_sensor - previous_sensor

            # Computes slope
            slope = float(sensor_dif) / time_dif

            for c in range(0, round(frames_in_between) + 1):
                sensor_list[sensor_index].append(previous_sensor + slope * c * time_between_frames)

        # Appends angle data
        for finger_index in range(0, NUM_FINGERS):
            for limb_index in range(0, NUM_LIMBS_PER_FINGER):
                # Retrieves relevant angle values
                previous_angle = smooth_old_angle_list[finger_index][limb_index][f - 1]
                current_angle = smooth_old_angle_list[finger_index][limb_index][f]

                # Computes difference
                angle_dif = current_angle - previous_angle

                # Computes slope
                slope = float(angle_dif) / time_dif

                # Appends the angles to the new list
                for c in range(0, round(frames_in_between) + 1):
                    angle_list[finger_index][limb_index].append(previous_angle + slope * c * time_between_frames)

    # Adds the missing front data
    time_list.insert(0, time_list[0])

    for sensor_index in range(0, sensor_index):
        sensor_list[finger_index].insert(0, sensor_list[finger_index][0])

    for finger_index in range(0, NUM_FINGERS):
        for limb_index in range(0, NUM_LIMBS_PER_FINGER):
            angle_list[finger_index][limb_index].insert(0, angle_list[finger_index][limb_index][0])

    # Goes through the function as many times as indicated
    return smooth_data(new_frame_rate, new_frame_rate,
                       time_list, sensor_list, angle_list,
                       times_remain=times_remain - 1)


# Post-processing to obtain limb angular velocity/acceleration
def generate_derivative_limb_data(original_list):
    derivative_list = []
    for index in range(1, len(original_list)):
        derivative_list.append(original_list[index] - original_list[index - 1])
    derivative_list.insert(0, derivative_list[0])  # assigns the first element to be that of the second
    return derivative_list


def post_data_processor(old_file_name, new_file_name, old_frame_rate, new_frame_rate, times_remain=1):
    # Obtains old file input
    reader = h5py.File("C:\\Git Data\\Virtual-Hand-Data\\training_datasets\\" + old_file_name + ".hdf5", 'r')

    old_time_list = float_int_unknownArray2list(reader.get("time"))
    old_sensor_list = float_int_unknownArray2list(reader.get("sensor"))
    old_angle_list = float_int_unknownArray2list(reader.get("angle"))

    reader.close()

    # Running average computations
    for sensor_index in range(0, NUM_SENSORS):
        old_sensor_list[sensor_index] = compute_forward_running_average(old_sensor_list[sensor_index],
                                                                        running_average=2)

    assert len(old_time_list) == len(old_sensor_list[0]) == len(old_angle_list[0][0])

    # Computes the smoothing
    print("Computing the smoothing...")
    time_list, sensor_list, angle_list = smooth_data(
        old_frame_rate=old_frame_rate,
        new_frame_rate=new_frame_rate,
        old_time_list=old_time_list,
        old_sensor_list=old_sensor_list,
        old_angle_list=old_angle_list,
        times_remain=times_remain)
    # Data to fill and return
    velocity_list = [[[] for b in range(0, NUM_LIMBS_PER_FINGER)] for a in range(0, NUM_FINGERS)]
    acceleration_list = [[[] for b in range(0, NUM_LIMBS_PER_FINGER)] for a in range(0, NUM_FINGERS)]

    for finger_index in range(0, NUM_FINGERS):
        for limb_index in range(0, NUM_LIMBS_PER_FINGER):
            # Calculated limb velocities based on the limb angles
            velocity_list[finger_index][limb_index] = generate_derivative_limb_data(
                angle_list[finger_index][limb_index])

            # Calculated limb accelerations based on the limb velocities
            acceleration_list[finger_index][limb_index] = generate_derivative_limb_data(
                velocity_list[finger_index][limb_index])

    # Just in case
    assert len(angle_list[0][0]) == len(velocity_list[0][0]) == len(acceleration_list[0][0]) \
           == len(time_list) == len(sensor_list[0])

    print("Saving the data...")

    # Saves the training data
    hf = h5py.File("C:\\Git Data\\Virtual-Hand-Data\\training_datasets\\" + new_file_name + ".hdf5", 'w')
    hf.create_dataset("time", data=time_list)
    hf.create_dataset("sensor", data=sensor_list)
    hf.create_dataset("angle", data=angle_list)
    hf.create_dataset("velocity", data=velocity_list)
    hf.create_dataset("acceleration", data=acceleration_list)
    hf.close()

    print("Data is saved.")


post_data_processor(old_file_name=ORIGINAL_FILE_NAME, new_file_name=NEW_FILE_NAME,
                    old_frame_rate=ORIGINAL_FRAME_RATE, new_frame_rate=NEW_FRAME_RATE,
                    times_remain=loop_times)

if manual_control:
    next_frame = "next"
    while next_frame == "next" or next_frame == "n":
        PlotData(training_name=NEW_FILE_NAME)

        next_frame = input("'next' frame? (any other input results in save) ")

        if next_frame == "next" or next_frame == "n":
            post_data_processor(old_file_name=NEW_FILE_NAME, new_file_name=NEW_FILE_NAME,
                                old_frame_rate=NEW_FRAME_RATE, new_frame_rate=NEW_FRAME_RATE,
                                times_remain=loop_times)

if draw_result:
    PlotData(training_name=NEW_FILE_NAME)
