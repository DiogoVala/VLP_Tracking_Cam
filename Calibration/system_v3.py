import io
import time
import threading
import queue
import picamera
from PIL import Image
from picamera.array import PiYUVArray
import datetime
from PIL import Image
import cv2
import numpy as np
import math
from scipy.spatial.transform import Rotation
from numpy.linalg import inv
from sys_calibration_bare import *
from sys_connection import *

# Run system calibration before starting camera
valid_markers, camera_pos, camera_ori, mapx, mapy, cameraMatrix, cameraDistortion, newCameraMatrix, rmat, tvec = runCalibration()
if(valid_markers == 0):
	print("Exiting program.")
	quit()

done = False # Global to indicate end of processing (To stop threads)
lock = threading.Lock() #
pool = [] # Pool of ImageProcessor threads
queue = queue.Queue()

# Camera Settings and Startup
RESOLUTION = (4032, 3040)
rescale_factor = 16
crop_window = 100

#RESOLUTION = (2016, 1520)
camera = picamera.PiCamera()
camera.resolution = RESOLUTION
#camera.exposure_mode = 'auto'
#camera.framerate = 90
#camera.iso 	= 1600
#camera.start_preview()
frame_T_ms=242 # Time between frames. Adjusted for frame acquisition delay
print("Camera warming up.")
time.sleep(1)

# Blob detector (High Resolution)
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 50
params.maxArea = 10000
params.minDistBetweenBlobs = 500
params.filterByCircularity = True
params.minCircularity = 0
params.filterByConvexity = True
params.minConvexity = 0
params.filterByInertia = True
params.minInertiaRatio = 0.1
detector_h = cv2.SimpleBlobDetector_create(params)

# Blob detector (Rescaled Resolution)
params_low = cv2.SimpleBlobDetector_Params()
params_low.filterByArea = True
params_low.minArea = int(params.minArea/rescale_factor)
params_low.maxArea = int(params.maxArea*rescale_factor)
params_low.minDistBetweenBlobs = int(params.minDistBetweenBlobs/rescale_factor)
params_low.filterByCircularity = True
params_low.minCircularity = 0
params_low.filterByConvexity = True
params_low.minConvexity = 0
params_low.filterByInertia = True
params_low.minInertiaRatio = 0.1
detector_l = cv2.SimpleBlobDetector_create(params_low)

# Color detection thersholds (YUV)
lower_range = np.array([  0,  0, 83])
upper_range = np.array([233,255,255])

# Returns (x,y) real world coordinates at height z.
def getWorldCoordsAtZ(image_point, z, mtx, rmat, tvec):

	camMat = np.asarray(mtx)
	iRot = inv(rmat.T)
	iCam = inv(camMat)

	uvPoint = np.ones((3, 1))

	# Image point
	uvPoint[0, 0] = image_point[0]
	uvPoint[1, 0] = image_point[1]

	tempMat = np.matmul(np.matmul(iRot, iCam), uvPoint)
	tempMat2 = np.matmul(iRot, tvec)

	s = (z + tempMat2[2, 0]) / tempMat[2, 0]
	wcPoint = np.matmul(iRot, (np.matmul(s * iCam, uvPoint) - tvec))

	# wcPoint[2] will not be exactly equal to z, but very close to it
	assert int(abs(wcPoint[2] - z) * (10 ** 8)) == 0
	wcPoint[2] = z

	return wcPoint


Nframes=0 # Number of processed frames
# Processing pipeline for each frame
def image_processor(frame):
	#print("Processing frame")

	global Nframes, done
	with lock:
		print("Acquired frames", Nframes)
		Nframes+=1

	if Nframes > 100:
		stop = datetime.datetime.now()
		elapsed = (stop - start).total_seconds()
		print("Framerate:",Nframes/elapsed)
		with lock:
			done=True

	total = datetime.datetime.now()

	# Resize high resolution to low resolution
	frame_low = cv2.resize(frame, (int(RESOLUTION[0]/rescale_factor),int(RESOLUTION[1]/rescale_factor)),interpolation = cv2.INTER_NEAREST)

	# Filter low resolution frame by color
	mask_low = cv2.inRange(frame_low, lower_range, upper_range)

	# Blob detector
	keypoints_low = detector_l.detect(mask_low)

	# Get rough LED position from low resolution mask
	if keypoints_low:
		leds_rough = [keypoint.pt for keypoint in keypoints_low]
		leds_rough = [(int(x)*rescale_factor, int(y)*rescale_factor) for x,y in leds_rough]

	# Crop frame around each LED
	leds_refined=[]
	for led in leds_rough:
		x=int(led[0])
		y=int(led[1])
		yuv_crop = frame[(y-crop_window):(y+crop_window), (x-crop_window):(x+crop_window)]
		mask = cv2.inRange(yuv_crop, lower_range, upper_range)

		# Look for blobs in each cropped region
		keypoints_high = detector_h.detect(mask)

		# Refine LED positions
		for keypoint_high in keypoints_high:
			led_refined = keypoint_high.pt
			led_refined = (round(led_refined[0])+x-crop_window, round(led_refined[1])+y-crop_window)
			leds_refined.append(led_refined)

	if leds_refined:
		for led in leds_refined:
			x=int(led[0])
			y=int(led[1])
			frame = cv2.line(frame, (x-30, y), (x+30, y),(0,0,255), 5)
			frame = cv2.line(frame, (x, y-30), (x, y+30),(0,0,255), 5)

	leds_refined = np.array(leds_refined, dtype=np.float32)

	tic = datetime.datetime.now()
	undistorted_coords = cv2.undistortPoints(leds_refined, cameraMatrix, cameraDistortion, None, newCameraMatrix)
	print("cv2.undistortPoints (s):",(datetime.datetime.now()-tic).microseconds/1000000)

	tic = datetime.datetime.now()
	realWorld_coords = []
	for coord in undistorted_coords:
		realWorld_coords.append(getWorldCoordsAtZ(coord[0], 0, cameraMatrix, rmat, tvec))
	print("getWorldCoordsAtZ (s):",(datetime.datetime.now()-tic).microseconds/1000000)

	print("Pixel coordinates:")
	for coord in undistorted_coords:
		print((coord[0][0], coord[0][1]))

	print("Real world coordinates:")
	for coord in realWorld_coords:
		coord.tolist()
		print((coord[0][0], coord[1][0]))
		
	print("Camera coordinates:", camera_pos)

	print("Total processing time (s):",(datetime.datetime.now()-total).microseconds/1000000)
	#queue.put(realWorld_coords)

	return

class ImageProcessor(threading.Thread):
	def __init__(self, processor_fcn):
		super(ImageProcessor, self).__init__()
		self.processor_fcn = processor_fcn
		self.stream = PiYUVArray(camera, size=RESOLUTION)
		self.event = threading.Event()
		self.terminated = False
		self.start()

	def run(self):
		# This method runs in a separate thread
		while not self.terminated:
			# Wait for an image to be written to the stream
			if self.event.wait(1):
				try:
					print(f"\n{threading.current_thread()} at: {datetime.datetime.now()}")
					self.stream.seek(0)
					frame = self.stream.array
					self.processor_fcn(frame) # Call function to process frame
				finally:
					self.stream.seek(0)
					self.stream.truncate()
					self.event.clear()
					with lock:
						pool.append(self)

# Generator of buffers for the capture_sequence method.
# Each buffer belongs to an ImageProcessor so each frame is sent to a different thread.
def streams():
	while not done:
		with lock:
			if pool:
				processor = pool.pop()
			else:
				processor = None
		if processor:
			yield processor.stream
			processor.event.set()
		else:
			# When the pool is starved, wait a while for it to refill
			break
			#time.sleep(0.1)
			
socker_sv = Socket_Server()
pool = [ImageProcessor(image_processor) for i in range(3)]
start = datetime.datetime.now()
prev_frame_time = datetime.datetime.now()
print("Starting capture.")
camera.capture_sequence(streams(), use_video_port=True, format='yuv')

print("Terminating program.")
while pool:
	with lock:
		processor = pool.pop()
	processor.terminated = True
	processor.join()
