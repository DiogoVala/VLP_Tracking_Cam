from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
import time
import cv2
import numpy as np

class PiStream:
	def __init__(self, src=0, resolution=(4032, 3040)):
		self.camera = PiCamera()
		self.camera.resolution = resolution
		self.camera.exposure_mode = 'auto'
		self.camera.iso 		  = 1600
		self.camera.framerate     = 10
		self.camera.start_preview()
		time.sleep(0.1)
		
		self.rawCapture = PiRGBArray(self.camera, size=resolution)
		self.frame = np.empty((resolution[1], resolution[0], 3), dtype=np.uint8) 
		
		self.stopped = False
		
	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self
		
	def update(self):
		if(self.stopped):
			return
		# keep looping infinitely until the thread is stopped
		self.camera.capture(self.rawCapture, format="rgb", use_video_port=True)
		self.frame=self.rawCapture.array
		self.rawCapture.truncate(0)

	def read(self):
		# return the frame most recently read
		return self.frame
		
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
