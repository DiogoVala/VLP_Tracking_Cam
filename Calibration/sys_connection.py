import socket
import threading

HOST = "192.168.91.11"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

class Socket_Server(threading.Thread):
    def __init__(self):
        print("Initiating socket server")
        self.terminated = False
        self.s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((HOST, PORT))
        self.s.listen()
        self.conn, self.addr = self.s.accept()
        print(f"Connected by client {self.addr}")
	
    def run(self):
        with self.conn:
            while not self.terminated:
                self.rxdata = conn.recv(1024)
                if not self.rxdata:
                    self.terminated = True
                else:
                    print("Received:", eval(self.rxdata))
                    self.data = eval(self.rxdata)
                    conn.sendall(data) # Reply with same data
                    self.event.set() # Set event signal on data acquisition
            self.s.close()
	    
class Socket_Client(threading.Thread):
    def __init__(self):
        self.s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((HOST, PORT))
        self.txdata=None
        self.terminated = False
	
    def run(self):
        while not self.terminated:
            if self.event.wait(1):
                try:
                    print("Sending:", str.encode(str(self.txdata)))
                    self.s.sendall(str.encode(str(self.txdata)))
                except:
                    print("Could not send data to server.")
                    self.s.close()
                    self.terminated = True
                finally:
                    self.event.clear()
        s.close()
