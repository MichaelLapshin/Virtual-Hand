"""
[SensorListener.py]
@description: API for receiving data from COM3 port.
@author: Michael Lapshin
"""

import serial
import time
import threading

print("Imported the SensorListener.py class successfully.")


class SensorReadingsListener(threading.Thread):
    buffer = ""  # Oldest element is first
    sensorReadings = {}
    wait4readings = True

    def __init__(self):
        threading.Thread.__init__(self)  # calls constructor of the Thread class
        try:
            self.port = serial.Serial('COM3', 9600, timeout=0)  # for COM3
        except:
            print("Warning, was not able to establish communications with COM3 port.")

    def start_thread(self):
        self.start()

    # Simple quit function for the thread
    def quit(self):
        self.close()

    # Once the thread starts, continuously read from the port and add any data to the buffer
    def run(self):
        try:
            ser_bytes = self.port.readline()
            time.sleep(1)
            inp = ser_bytes.decode('uft-8')
            self.buffer.append(inp)  # adds to the end of the queue
        except:
            print("Could not read from COM3 port.")

        # Adds all sensor data accumulated in the buffer to the dictionary
        raw_buffer_data = self.buffer.split(" ")
        if len(raw_buffer_data) > 0:
            for index in range(0, len(int(raw_buffer_data) / 2) - 1):
                if self.sensorReadings.has_keys(raw_buffer_data[index * 2]):
                    self.sensorReadings.add(raw_buffer_data[index * 2], int(raw_buffer_data[index * 2 + 1]))
                else:
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
