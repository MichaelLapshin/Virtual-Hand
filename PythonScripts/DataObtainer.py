"""
[DataObtainer.py]
#description: Script for obtaining finger angles and sensor readings.
@author: Michael Lapshin
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # To remove the redundant warnings
import time
import SensorListener
import MediapipeHandAngler
import numpy
import h5py

# Listeners declaration
sensor_data = SensorListener.SensorReadingsListener()
hand_angler = MediapipeHandAngler.HandAngleReader(framerate=30, resolution=480)

# Starts the listeners' threads
hand_angler.start_thread()
sensor_data.start_thread()
zeros = {}
print("B")

# Obtains zeroing information
zeroing_delay = int(input(
    "How much setup time do you need before starting to zero the sensor readings? (Zeroing timer starts after you enter the input)"))
time.sleep(zeroing_delay)
# Zeros the sensor data
zeros = None
while zeros is None:
    zeros = sensor_data.get_readings_frame()
sensor_data.wait4new_readings()

# Obtains training information
name_confirm = False
while not name_confirm:
    training_name = input("What is the name of the training?")
    # Figures out the name of the new training set
    name_exists = False
    for file_name in os.listdir("./training_datasets"):
        if training_name == file_name:
            name_exists = True
    if not name_exists:
        name_confirm = True
    else:
        name_confirm = str(input("That name already exists. Override file? (yes/no)")).lower() == "yes"

seconds_training = int(input("How many seconds of training do you want?"))
training_framerate = int(input("What is your desired framerate (FPS)?"))
training_delay = int(input(
    "How much setup time do you need before starting to store the data? (Starts timer starts after you enter the input)"))

time.sleep(max(training_delay-3,0))
print("Starting the training in")
print("3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")
time.sleep(1)
print("The training has begun!")
"""
# Training data format
time | sensor | angle | velocity | acceleration

time: time in nanoseconds since the start of the training sequence
sensor: [sensor1,sensor2,...,sensorN]
    - zeroed-sensor readings
angle: [[fingerAngularVelocities]*fingers]
    - angles of the finger limbs
velocity: [[fingerAngularVelocities]*fingers]
    - velocity of the finger limbs
acceleration: [[fingerAngular acceleration]*fingers] 
    - acceleration of the finger limbs
* Note: the angle(and its derivatives) lists is in the 5 by 3 by N format
"""

# Starts the recording

# Dictionaries
time_list = []
sensor_list = []
angle_list = [[[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]]
velocity_list = [[[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]]
acceleration_list = [[[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]]

# Indexes the incoming sensor data (sensor key character -> number between 0 and total sensor count)
sensor_to_index_map = {}
key_list = sensor_data.get_key_list()
key_index = 0
for key in key_list:
    sensor_to_index_map[key] = key_index
    key_index += 1
    sensor_list.append([])

# The data gathering
zero_time_ns = time.time_ns()
for frame_num in range(0, seconds_training * training_framerate):
    # Halts the program until it is time to take the next frame of the training data
    while time.time_ns() - zero_time_ns < 1000000000/training_framerate * frame_num:
        time.sleep(0.01)

    current_sensor_data = None
    # Waits until new sensor data is available
    while current_sensor_data is None:
        current_sensor_data = sensor_data.get_readings_frame()

    # Stores the data
    # Adds sensor data
    for key in current_sensor_data.keys():
        sensor_list[sensor_to_index_map[key]].append(current_sensor_data[key] - zeros[key])

    sensor_data.wait4new_readings()
    # Adds limb angle data
    limb_data = hand_angler.get_all_limb_angles()
    for finger_index in range(0, 5):
        for limb_index in range(0, 3):
            angle_list[finger_index][limb_index].append(limb_data[finger_index][limb_index])
    # Adds time data
    time_list.append(time.time_ns() - zero_time_ns)

print("The training sequence is now complete.")

# Closes ports and connections
sensor_data.quit()
hand_angler.quit()


# Post-processing to obtain limb angular velocity/acceleration
def generate_derivative_limb_data(original_list):
    derivative_list = []
    for index in range(1, len(original_list)):
        derivative_list.append(original_list[index] - original_list[index])
    derivative_list.insert(0, derivative_list[0])  # assigns the first element to be that of the second
    return derivative_list


for finger_index in range(0, 5):
    for limb_index in range(0, 3):
        # Calculated limb velocities based on the limb angles
        velocity_list[finger_index][limb_index] = generate_derivative_limb_data(angle_list[finger_index][limb_index])
        # Calculated limb accelerations based on the limb velocities
        acceleration_list[finger_index][limb_index] = generate_derivative_limb_data(velocity_list[finger_index][limb_index])

# Saves the training data
hf = h5py.File("./training_datasets/" + training_name + ".hdf5", 'w')
hf.create_dataset("time", data=time_list)
hf.create_dataset("sensor", data=sensor_list)
hf.create_dataset("angle", data=angle_list)
hf.create_dataset("velocity", data=velocity_list)
hf.create_dataset("acceleration", data=acceleration_list)
hf.close()
