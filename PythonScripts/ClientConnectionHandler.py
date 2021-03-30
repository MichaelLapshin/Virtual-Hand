"""
[ClientConnectionHandler.py]
@description: Class for establishing and interacting with external programs.
@author: Michael Lapshin
"""

import socket
import threading
import time
import select


class ClientConnectionHandler:
    running = True
    input_buffer = ""

    def __init__(self, HOST="127.0.0.1", PORT=5000):
        self.HOST = HOST
        self.PORT = PORT

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # establishes default connection
        # self.s.settimeout(value=30)
        self.s.connect((self.HOST, self.PORT))

        f = open("C:\\Git\\Virtual-Hand\\PythonScripts\\PythonClientInput.txt", 'w')
        f.write("")
        f.close()
        f = open("C:\\Git\\Virtual-Hand\\PythonScripts\\PythonClientOutput.txt", 'w')
        f.write("")
        f.close()

        self.input_thread = threading.Thread(target=self.data_receiver_thread_method)
        self.input_thread.start()

    def data_receiver_thread_method(self):
        """
        Function that continuously runs on the client connection handler thread to receive input.
        :return: void
        """

        while self.running:
            # ready = select.select(self.s, [], [], 30)
            # if ready[0]:
            self.input_buffer += self.s.recv(2048).decode("utf-8")  # todo, make sure it receives newlines
            # self.println("Received")
            if len(self.input_buffer) > 0:
                f = open("C:\\Git\\Virtual-Hand\\PythonScripts\\PythonClientInput.txt", 'a')
                f.write("===========\n" + self.input_buffer)
                f.close()

    def println(self, message):
        f = open("C:\\Git\\Virtual-Hand\\PythonScripts\\PythonClientOutput.txt", 'a')
        f.write("===========\n" + message + "\n")
        f.close()
        with self.s:
            self.s.send((message + "\n").encode("utf-8"))

    def input(self):
        while self.running:
            message_list = self.input_buffer.split("\n")

            if len(message_list) > 1:
                # Removes the message from the buffer and returns it
                message = message_list[0]
                self.input_buffer = self.input_buffer[len(message) + 1:]
                return message
            time.sleep(0.005)

    def stop(self):
        self.running = False
        self.s.close()
