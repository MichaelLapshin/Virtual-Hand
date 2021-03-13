"""
[SensorListener.py]
#description: API for receiving data from COM3 port.
@author: Michael Lapshin
"""

import serial
import time
import threading

print("Imported the SensorListener.py class successfully.")


class SensorListener(threading.Thread):
    buffer = ""  # Oldest element is first

    def __init__(self):
        threading.Thread.__init__(self)
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
            print(inp)  # todo, remove this when you're done with testing
            self.buffer.append(inp)  # adds to the end of the queue
        except:
            print("Could not read from COM3 port.")

    # nextData()
    # @return [sensorID#, sensorReading]
    #     [0,0] is a null-no data return
    def next_data(self):
        # If the buffer has accumulated enough data for a sensor, then it returns that value as requested
        if len(self.buffer.split(" ")) > 2:
            sensor, reading = self.buffer.split(" ")[0:1]
            self.buffer.lstrip(sensor + " " + reading + " ")  # removes data point from the buffer
            return [sensor, int(reading)]
        else:
            return [0, 0]
