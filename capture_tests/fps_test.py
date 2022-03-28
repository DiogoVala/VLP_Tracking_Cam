
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import datetime


resolution=(4032, 3040)
resolution=(1920, 1080)

camera = PiCamera()
camera.resolution = resolution
camera.exposure_mode = 'auto'
camera.framerate = 30
time.sleep(0.1)

rawCapture = PiRGBArray(camera, size=resolution)

Nframes=0
start = datetime.datetime.now()
for RGBArray in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
	Nframes+=1;
	frame = RGBArray.array
	rawCapture.truncate(0)
	if Nframes == 10:
		break

stop = datetime.datetime.now()
elapsed = (stop - start).total_seconds()
	
framerate=Nframes/elapsed

print(framerate)
