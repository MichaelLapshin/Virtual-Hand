"""
[ServerConnectionHandler.py]
@description: Used to connect one of more client connections. Relays the data between the clients.
@author: Michael Lapshin
"""

import socket
import threading
import time

# For debugging
print_to_logs = False


def controlled_print(message):
    if print_to_logs:
        print(message)


# Constants
INPUT_PORT = 5000  # the port through which the server receives information
OUTPUT_PORT = 6000  # the port through which the server sends information to the clients

# Generates basic connection
# Input socket
input_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
input_socket.bind(('', INPUT_PORT))
input_socket.listen(5)
print("Input socket successfully created. Port:", INPUT_PORT)

# Output socket
output_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
output_socket.bind(('', OUTPUT_PORT))
output_socket.listen(5)
print("Output socket successfully created. Port:", OUTPUT_PORT)

clients = []
running = True


class ClientHandler(threading.Thread):
    input_client_buffer = ""

    def __init__(self, input_connection, input_client_address, output_connection, output_client_address):
        threading.Thread.__init__(self)

        self.input_connection = input_connection
        self.input_client_address = input_client_address

        self.output_connection = output_connection
        self.output_client_address = output_client_address

        self.lock = threading.Lock()

        self.start()

    def run(self):
        global running

        while running:

            try:
                recv_input = self.input_connection.recv(1024).decode("utf-8")

                self.lock.acquire()
                self.input_client_buffer += recv_input
                self.lock.release()

            except Exception as e:
                print("ERROR...")
                print(e)
                if e == "timed out":
                    running = False
                print(self.input_connection)
                print(self.input_client_buffer)

            time.sleep(0.003)

    def get_buffer_line(self):
        self.lock.acquire()

        buffer_list = self.input_client_buffer.lstrip("$ ").rstrip(" ").split('$')
        line = ""
        if len(buffer_list) > 1:
            line = buffer_list[0]
            self.input_client_buffer = self.input_client_buffer.lstrip("$ ").rstrip(" ")[len(line) + 1:]

        self.lock.release()

        if len(line) > 0:
            return line

        return None

    def send(self, message):
        self.output_connection.send(message)
        time.sleep(0.003)

    def get_address(self):
        return self.input_client_address

    def stop(self):
        self.input_connection.close()
        self.output_connection.close()


# Connections accepter
def connection_establisher():
    global running, clients, input_socket

    # If client connects, then add it to the clients list
    while running:
        input_connection, input_client_address = input_socket.accept()
        time.sleep(0.5)
        output_connection, output_client_address = output_socket.accept()
        time.sleep(0.5)

        input_connection.settimeout(12.0)
        output_connection.settimeout(12.0)

        input_connection.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 20000, 20000))
        output_connection.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 20000, 20000))

        clients.append(ClientHandler(input_connection, input_client_address, output_connection, output_client_address))
        print("Established connection with... in:" + str(input_client_address) + " : out:" + str(output_client_address))


connection_thread = threading.Thread(target=connection_establisher, name="Connection Thread")
connection_thread.start()


# Server console thread
def console_threader():
    global running, clients, data

    while running:
        inp = input()
        if inp.lower() == "stop" or inp.lower() == "quit" or inp.lower() == "end":
            running = False
        elif inp.lower() == "connections":
            print("===== Connection Information =====")
            print("Number of connections: " + len(clients))
        else:
            print("Unknown command.")
        time.sleep(1)


console_thread = threading.Thread(target=console_threader, name="Console Thread")
console_thread.start()

# Server logic
print("Threads successfully created.")
while running:
    for client in clients:
        message = client.get_buffer_line()

        if message is not None:
            message += "$"
            controlled_print("Sending over \"" + message + "\" from " + str(client.get_address()) + " to " +
                             str(len(clients) - 1) + " other connections.")

            for other_client in clients:
                if other_client.get_address() != client.get_address():
                    controlled_print(
                        "Sending " + str(message.encode("utf-8")) + " to " + str(other_client.get_address()))
                    other_client.send(message.encode("utf-8"))

# Ends the server program
print("Stopping the server program.")
time.sleep(0.5)  # todo, change this back to 0.5
for client in clients:
    client.stop()
# input_socket.shutdown(socket.SHUT_RDWR)
input_socket.close()
print("Successfully ended the server program.")
