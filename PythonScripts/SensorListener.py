"""
[SensorListener.py]
@description: API for receiving data from COM3 port.
@author: Michael Lapshin
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # To remove the redundant warnings
import serial
import time
import threading
import subprocess

print("Imported the SensorListener.py class successfully.")


class SensorReadingsListener(threading.Thread):
    buffer = ""  # Oldest element is first
    sensorReadings = {}
    wait4readings = True
    running = True

    def __init__(self):
        threading.Thread.__init__(self)  # calls constructor of the Thread class
        self.daemon = True
        try:
            self.port = serial.Serial('COM3', 9600, timeout=100)  # for COM3
            time.sleep(0.5)
        except Exception as e:
            print("Warning, was not able to establish communications with COM3 port.")
            print("Error: ", e)
            if self.port.isOpen():
                print("Closed the port.")
                self.port.close()

    def start_thread(self):
        self.start()

    # Simple quit function for the thread
    def quit(self):
        self.running = False
        time.sleep(3)
        if self.port.isOpen():
            self.port.close()

    # Once the thread starts, continuously read from the port and add any data to the buffer
    def run(self):
        while self.running:

            if self.port.inWaiting() > 0:  # len(next_line) > 0:
                next_char = self.port.read().decode("utf-8")
                if next_char == "\r":
                    next_char = " "
                elif next_char == "\n":
                    next_char = ""

                self.buffer += next_char

                # Adds all sensor data accumulated in the buffer to the dictionary
                raw_buffer_data = self.buffer.split(" ")
                if len(raw_buffer_data) > 0 and raw_buffer_data[0] == "":
                    raw_buffer_data.pop(0)
                if len(raw_buffer_data) > 0:
                    for index in range(0, max(int(len(raw_buffer_data) / 2) - 1, 0)):
                        self.sensorReadings[raw_buffer_data[index * 2]] = int(raw_buffer_data[index * 2 + 1])

                    # Checks if the dictionary is filled so that it can be used
                    all_keys_have_values = True
                    for readingKey in self.sensorReadings.keys():
                        if self.sensorReadings.get(readingKey) is None:
                            all_keys_have_values = False
                            break
                    if all_keys_have_values:
                        self.wait4readings = False

    # Returns unique keys, to be used after the system is setup for accuracy
    def get_key_list(self):
        return self.sensorReadings.keys()

    # Getter for the batch of sensor readings list
    def get_readings_frame(self):
        if not self.wait4readings:
            return self.sensorReadings
        return None

    # Adds a tag that will not allow data to return if the data set is not complete
    def wait4new_readings(self):
        for readingKey in self.sensorReadings.keys():
            self.sensorReadings[readingKey] = None
        self.wait4readings = True

    def print_raw_sensor_readings(self):
        # print("=== Sensor Readings ===")
        for key in self.sensorReadings.keys():
            print(key, self.sensorReadings.get(key))
