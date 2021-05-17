"""
[ClientConnectionHandler.py]
@description: Class for establishing and interacting with external programs.
@author: Michael Lapshin
"""

import socket
import threading
import time
import select
import sys

sys.stderr = open("C:\\Git\\Virtual-Hand\\PythonScripts\\PythonClientError_ConnectionHandler.txt", "w")
sys.stdout = open("C:\\Git\\Virtual-Hand\\PythonScripts\\PythonClientLog.txt", 'a')

print_to_logs = False

def controlled_print(message):
    """
    A print statement that can be turned off with single variable "print_to_logs".
    Used to include/remove less important information.
    """
    if print_to_logs:
        print(str(message))


class ClientConnectionHandler:
    running = True
    input_buffer = ""
    SOCKET_TIMEOUT_SECONDS = 30.0

    def __init__(self, HOST="127.0.0.1", INPUT_PORT=6000, OUTPUT_PORT=5000):
        self.HOST = HOST
        self.INPUT_PORT = INPUT_PORT
        self.OUTPUT_PORT = OUTPUT_PORT

        self.output_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # establishes default connection
        self.output_socket.setsockopt(socket.SOL_SOCKET, socket.TCP_NODELAY, True)
        self.output_socket.setblocking(True)
        self.output_socket.connect((self.HOST, self.OUTPUT_PORT))
        time.sleep(0.5)

        self.input_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # establishes default connection
        self.input_socket.setblocking(True)
        self.input_socket.setsockopt(socket.SOL_SOCKET, socket.TCP_NODELAY, True)
        self.input_socket.connect((self.HOST, self.INPUT_PORT))
        time.sleep(0.5)

        self.input_socket.settimeout(ClientConnectionHandler.SOCKET_TIMEOUT_SECONDS)
        self.output_socket.settimeout(ClientConnectionHandler.SOCKET_TIMEOUT_SECONDS)

        self.lock = threading.Lock()

        self.input_thread = threading.Thread(target=self.data_receiver_thread_method)
        self.input_thread.start()

    def data_receiver_thread_method(self):
        """
        Function that continuously runs on the client connection handler thread to receive input.
        :return: void
        """

        while self.running:
            # Receives input
            recv_input = self.input_socket.recv(1024).decode("utf-8")

            # Locks and modifies the buffer
            self.lock.acquire()

            self.input_buffer += recv_input
            if len(self.input_buffer) > 0:
                controlled_print("Current buffer: " + self.input_buffer)

            self.lock.release()

            time.sleep(0.001)

    def print(self, message):
        message = str(message)
        controlled_print("Sending over to C#: " + message)

        self.output_socket.send((message + "$").encode("utf-8"))
        time.sleep(0.001)

    def input(self):
        while self.running:
            self.lock.acquire()
            message_list = self.input_buffer.lstrip("$ ").split("$")
            message = ""

            if len(message_list) > 1:
                # Removes the message from the buffer and returns it
                message = message_list[0]
                self.input_buffer = self.input_buffer.lstrip("$ ")[len(message) + 1:]
                controlled_print("Received: " + message + "               New buffer is: " + self.input_buffer)
            self.lock.release()

            if len(message) > 0:
                return message

            time.sleep(0.001)

    def stop(self):
        self.running = False
        self.input_socket.close()
        self.output_socket.close()
