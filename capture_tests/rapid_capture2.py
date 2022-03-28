
import io
import time
import threading
import picamera
import datetime
from PIL import Image

done = False # Global to indicate end of processing (To stop threads)
lock = threading.Lock() # 
pool = []
Nframes=0

def image_processor(frame):
    global Nframes, done
    Nframes+=1
    if Nframes == 10000:
        stop = datetime.datetime.now()
        elapsed = (stop - start).total_seconds()
        print(Nframes/elapsed)
        done=True
        
stream = io.BytesIO()

with picamera.PiCamera() as camera:

    camera.resolution = (4032, 3040)
    camera.resolution = (1280, 720)
    camera.framerate = 5
    time.sleep(2)
    start = datetime.datetime.now()
    camera.capture_sequence(stream, use_video_port=True)
    
    stream.seek(0)
    frame = Image.open(stream) # Grab frame from stream
    image_processor(frame) # Call function to process frame

