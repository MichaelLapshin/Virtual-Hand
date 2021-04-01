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

sys.stderr = open("PythonClientError_ConnectionHandler.txt", "w")

class ClientConnectionHandler:
    running = True
    input_buffer = ""

    def __init__(self, HOST="127.0.0.1", PORT=5000):
        self.HOST = HOST
        self.PORT = PORT

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # establishes default connection
        # self.s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        # self.s.setblocking(False)
        self.s.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 20000, 20000))
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
            self.input_buffer += self.s.recv(256).decode("utf-8")  # todo, make sure it receives newlines
            # self.println("Received")
            if len(self.input_buffer) > 0:
                f = open("C:\\Git\\Virtual-Hand\\PythonScripts\\PythonClientInput.txt", 'a')
                f.write("===========\n" + self.input_buffer)
                f.close()
            # time.sleep(0.01)

    def println(self, message):
        f = open("C:\\Git\\Virtual-Hand\\PythonScripts\\PythonClientOutput.txt", 'a')
        f.write("===========\n" + message + "\n")
        f.close()
        with self.s:
            self.s.send((message + "\n").encode("utf-8"))
        time.sleep(0.01)

    def input(self):
        while self.running:
            message_list = self.input_buffer.lstrip("\n ").rstrip(" ").split("\n")

            for m in message_list:
                if (len(message_list) > 1 and m == ""):
                    f = open("C:\\Git\\Virtual-Hand\\PythonScripts\\PythonClientOutput.txt", 'a')
                    f.write("===========\n" + "AAAAAAAAAAAAAAAAA "+message_list[1] + "\n")
                    for k in message_list:
                        f.write(k+"\n")
                    f.close()

            if len(message_list) > 1:
                # Removes the message from the buffer and returns it
                message = message_list[0]
                # self.input_buffer = self.input_buffer.lstrip("\n ").rstrip(" ")[len(message) + 1:]
                self.input_buffer = message_list[1]
                return message
            time.sleep(0.005)

    def stop(self):
        self.running = False
        self.s.close()
