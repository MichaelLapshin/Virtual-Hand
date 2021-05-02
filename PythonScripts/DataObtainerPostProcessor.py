"""
[DataObtainerPostProcessor.py]
@description: Script for processing the training data (eg. smoothing the data, velocity, acceleration, etc.)
@author: Michael Lapshin
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # To remove the redundant warnings
import time
import SensorListener
import MediapipeHandAngler
import numpy
import h5py
import copy
import math

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

ORIGINAL_FILE_NAME = input("Original file name: ")
ORIGINAL_FRAME_RATE = int(input("Original dataset framerate: "))
NEW_FRAME_RATE = int(input("New frame rate (for data interpolation): "))
SMOOTH_FRAMES = int(input("How many frames far should the smoothing look? "))  # 5 frames?
SMOOTH_DIF_THRESHOLD = float(
    input("What degree difference in the frames should the smooth?")) * 180.0 / math.pi  # 5 degrees?

reader = h5py.File("./training_datasets/" + ORIGINAL_FILE_NAME + ".hdf5", 'r')
NEW_FILE_NAME = ORIGINAL_FILE_NAME.rstrip("_raw") + "_smoothed"


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


original_time_list = float_int_unknownArray2list(reader.get("time"))
original_sensor_list = float_int_unknownArray2list(reader.get("sensor"))
original_angle_list = float_int_unknownArray2list(reader.get("angle"))

assert len(original_time_list) == len(original_sensor_list[0]) == len(original_angle_list[0][0])

# Empty lists to be filled by the program
NUM_FINGERS = 5
NUM_LIMBS_PER_FINGER = 3
time_list = []
sensor_list = [[] for a in range(0, NUM_FINGERS)]
angle_list = [[[] for b in range(0, NUM_LIMBS_PER_FINGER)] for a in range(0, NUM_FINGERS)]
velocity_list = [[[] for b in range(0, NUM_LIMBS_PER_FINGER)] for a in range(0, NUM_FINGERS)]
acceleration_list = [[[] for b in range(0, NUM_LIMBS_PER_FINGER)] for a in range(0, NUM_FINGERS)]

# Smooth the angular data (averages out the angle using nearby samples)
smooth_original_angle_list = [[[] for b in range(0, NUM_LIMBS_PER_FINGER)] for a in range(0, NUM_FINGERS)]
for finger_index in range(0, NUM_FINGERS):
    for limb_index in range(0, NUM_LIMBS_PER_FINGER):
        for f in range(0, len(original_time_list)):
            new_angle = 0
            frames_considered = 0

            # To the right of the subject frame
            for r in range(1, SMOOTH_FRAMES):
                if f + r >= len(original_angle_list[finger_index][limb_index]):
                    break
                if math.fabs(original_angle_list[finger_index][limb_index][f + r] -
                             original_angle_list[finger_index][limb_index][f]) > SMOOTH_DIF_THRESHOLD:
                    pass
                else:
                    new_angle += original_angle_list[finger_index][limb_index][f + r]
                    frames_considered += 1

            # To the left of the subject frame
            for l in range(0, SMOOTH_FRAMES):
                if f - l < 0:
                    break
                if math.fabs(original_angle_list[finger_index][limb_index][f - l] -
                             original_angle_list[finger_index][limb_index][f]) > SMOOTH_DIF_THRESHOLD:
                    pass
                else:
                    new_angle += original_angle_list[finger_index][limb_index][f - l]
                    frames_considered += 1

            if frames_considered == 0:
                new_angle = original_angle_list[finger_index][limb_index][f]
                frames_considered = 1

            smooth_original_angle_list[finger_index][limb_index].append(new_angle / frames_considered)

# Adjust the frame rate (linear interpolation)
original_ms_per_frame = 1000.0 / ORIGINAL_FRAME_RATE
new_ms_per_frame = 1000.0 / NEW_FRAME_RATE

for f in range(1, len(original_time_list)):
    # Retrieves relevant time values
    previous_time = original_time_list[f - 1]
    current_time = original_time_list[f]

    # Computes difference
    time_dif = current_time - previous_time
    assert time_dif > 0

    # Computes how many frames and the time difference between each frame to be used
    frames_in_between = max(1, round((time_dif * 1.0) / new_ms_per_frame - 1))  # rounds number
    time_between_frames = (time_dif * 1.0) / frames_in_between

    # Appends time data
    for c in range(0, frames_in_between + 1):
        time_list.append(int(current_time - time_between_frames * c))

    # Appends sensor data
    for finger_index in range(0, NUM_FINGERS):
        # Retrieves relevant sensor values
        previous_sensor = original_sensor_list[finger_index][f - 1]
        current_sensor = original_sensor_list[finger_index][f]

        # Computes difference
        sensor_dif = previous_sensor - current_sensor

        # Computes slope
        slope = sensor_dif / time_dif

        for c in range(0, frames_in_between + 1):
            sensor_list[finger_index].append(current_sensor - slope * c * time_between_frames)

    # Appends angle data
    for finger_index in range(0, NUM_FINGERS):
        for limb_index in range(0, NUM_LIMBS_PER_FINGER):
            # Retrieves relevant angle values
            previous_angle = smooth_original_angle_list[finger_index][limb_index][f - 1]
            current_angle = smooth_original_angle_list[finger_index][limb_index][f]

            # Computes difference
            angle_dif = previous_angle - current_angle

            # Computes slope
            slope = angle_dif / time_dif

            # Appends the angles to the new list
            for c in range(0, frames_in_between + 1):
                angle_list[finger_index][limb_index].append(current_angle - slope * c * time_between_frames)

# Adds the missing front data
time_list.insert(0, time_list[0])
for finger_index in range(0, NUM_FINGERS):
    sensor_list[finger_index].insert(0, sensor_list[finger_index][0])
    for limb_index in range(0, NUM_LIMBS_PER_FINGER):
        angle_list[finger_index][limb_index].insert(0, angle_list[finger_index][limb_index][0])


# Post-processing to obtain limb angular velocity/acceleration
def generate_derivative_limb_data(original_list):
    derivative_list = []
    for index in range(1, len(original_list)):
        derivative_list.append(original_list[index] - original_list[index - 1])
    derivative_list.insert(0, derivative_list[0])  # assigns the first element to be that of the second
    return derivative_list


for finger_index in range(0, NUM_FINGERS):
    for limb_index in range(0, NUM_LIMBS_PER_FINGER):
        # Calculated limb velocities based on the limb angles
        velocity_list[finger_index][limb_index] = generate_derivative_limb_data(angle_list[finger_index][limb_index])
        # Calculated limb accelerations based on the limb velocities
        acceleration_list[finger_index][limb_index] = generate_derivative_limb_data(
            velocity_list[finger_index][limb_index])

# Just in case
assert len(angle_list[0][0]) == len(velocity_list[0][0]) == len(acceleration_list[0][0]) \
       == len(time_list) == len(sensor_list[0])

print("Saving the data...")

# Saves the training data
hf = h5py.File("./training_datasets/" + NEW_FILE_NAME + ".hdf5", 'w')
hf.create_dataset("time", data=time_list)
hf.create_dataset("sensor", data=sensor_list)
hf.create_dataset("angle", data=angle_list)
hf.create_dataset("velocity", data=velocity_list)
hf.create_dataset("acceleration", data=acceleration_list)
hf.close()

print("All done.")
