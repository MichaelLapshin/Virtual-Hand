"""
[ServerConnectionHandler.py]
@description: Used to connect one of more client connections. Relays the data between the clients.
@author: Michael Lapshin
"""

import socket
import threading
import time

# Constants
PORT = 5000

# Generates basic connection
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('', PORT))
server_socket.listen(5)
# server_socket.settimeout(value=60)
print("Server successfully created. Bound to port:", PORT)
clients = []
running = True

class ClientHandler(threading.Thread):
    client_buffer = ""

    def __init__(self, connection, client_address):
        threading.Thread.__init__(self)

        self.connection = connection
        self.client_address = client_address
        # self.lock = threading.Lock()

        self.start()

    def run(self):
        while running:
            # self.lock.acquire()
            # byte_data = "".encode("utf-8")
            try:
                # byte_data = self.connection.recv(2048)
                # print("byte data:", byte_data)
                self.client_buffer += self.connection.recv(2048).decode("utf-8")
            except Exception as e:
                print("ERROR...")
                print(e)
                print(self.client_address)
                print(self.client_buffer)
            # self.lock.release()
            time.sleep(0.005)

            # if self.client_buffer == "Received":  # todo, problem here?
            #     self.client_buffer = ""
            #     print(self.client_address, "received the message.")

    def get_buffer_line(self):
        if len(self.client_buffer.split('\n')) > 1:
            message = self.client_buffer.split('\n')[0]
            # self.client_buffer = self.client_buffer[len(message) + 1:]
            self.client_buffer = self.client_buffer.split('\n')[1]
            return message
        return None

    def send(self, message):
        # self.lock.acquire()
        self.connection.send(message)
        # self.lock.release()

    def get_address(self):
        return self.client_address

    def stop(self):
        self.connection.close()


# Connections accepter
def connection_establisher():
    global running, clients, server_socket, lock

    # If client connects, then add it to the clients list
    while running:
        connection, client_address = server_socket.accept()
        # clients[client_address] = connection
        clients.append(ClientHandler(connection, client_address))
        print("Established connection with", client_address)


connection_thread = threading.Thread(target=connection_establisher, name="Connection Thread")
threading.Thread()
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
            print("Number of connections: ", len(clients))
        else:
            print("Unknown command.")


console_thread = threading.Thread(target=console_threader, name="Console Thread")
console_thread.start()

# Server logic
print("Threads successfully created.")
while running:
    for client in clients:
        message = client.get_buffer_line()

        if message is not None:
            print("Sending over \"" + message + "\" from " + str(client.get_address()) + " to",
                  len(clients) - 1, "other connections.")

            for other_client in clients:
                if other_client.get_address() != client.get_address():
                    print("Sending", (message+"\n").encode("utf-8"), "to", other_client.get_address())
                    other_client.send((message+"\n").encode("utf-8"))

# Ends the server program
print("Stopping the server program.")
time.sleep(0.5)  # todo, change this back to 0.5
for client in clients:
    client.stop()
server_socket.shutdown(socket.SHUT_RDWR)
server_socket.close()
print("Successfully ended the server program.")
