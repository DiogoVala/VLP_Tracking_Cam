import socket

HOST = "192.168.91.11"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

class Socket_Server(threading.Thread):
    def __init__(self):
        self.terminated = False
        self.s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        print(f"Connected by client {addr}")
	
    def run(self):
        with conn:
            while not self.terminated::
                self.rxdata = conn.recv(1024)
                if not self.rxdata:
                    self.terminated = True
                else:
                    self.data = eval(self.rxdata)
                    conn.sendall(data) # Reply with same data
                    self.event.set() # Set event signal on data acquisition
            s.close()
	    
class Socket_Client(threading.Thread):
    def __init__(self):
        self.s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))
        self.txdata=None
        self.terminated = False
	
    def run(self):
        while not self.terminated:
            if self.event.wait(1):
                try:
                    s.sendall(str.encode(str(self.txdata)))
                except:
                    print("Could not send data to server.")
                    s.close()
                    self.terminated = True
                finally:
                    self.event.clear()
        s.close()