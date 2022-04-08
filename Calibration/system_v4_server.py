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
from numpy import array, cross
from numpy.linalg import solve, norm

# Run system calibration before starting camera
valid_markers, camera_pos, camera_ori, mapx, mapy, cameraMatrix, cameraDistortion, newCameraMatrix, rmat, tvec = runCalibration()
if(valid_markers == 0):
	print("Exiting program.")
	quit()

# Thread managing stuff
lock = threading.Lock() # Interprocess variable for mutual exclusion
pool = [] # Pool of ImageProcessor threads

# Camera Settings and Startup
RESOLUTION = (4032, 3040)
rescale_factor = 16
crop_window = 100

#RESOLUTION = (2016, 1520)
camera = picamera.PiCamera()
camera.resolution = RESOLUTION
camera.exposure_mode = 'auto'
#camera.framerate = 90
camera.iso 	= 1600
#camera.start_preview()
frame_T_ms=242 # Time between frames. Adjusted for frame acquisition delay
print("Camera warming up.")
time.sleep(1)

# Blob detector settings (High Resolution)
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

# Blob detector settings (Rescaled Resolution)
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

# LED position data from this camera
this_cam_data=None

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

# Processing pipeline for each frame
def image_processor(frame):
	global this_cam_data

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
			leds_refined = np.array(leds_refined, dtype=np.float32)

			undistorted_coords = cv2.undistortPoints(leds_refined, cameraMatrix, cameraDistortion, None, newCameraMatrix)

			realWorld_coords = []
			for coord in undistorted_coords:
				realWorld_coords.append(getWorldCoordsAtZ(coord[0], 0, cameraMatrix, rmat, tvec))														

			for coord in realWorld_coords:
				coord.tolist()
				this_cam_data=[(coord[0][0], coord[1][0]), (camera_pos[0][0],camera_pos[1][0],camera_pos[2][0])]
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
					#print(f"\n{threading.current_thread()} at: {datetime.datetime.now()}")
					self.stream.seek(0)
					frame = self.stream.array
					self.processor_fcn(frame) # Call function to process frame
				finally:
					self.stream.seek(0)
					self.stream.truncate()
					self.event.clear()
					with lock:
						pool.append(self)

def intersect(other_cam_data):
	try:
		P0=np.array([[this_cam_data[1][0], this_cam_data[1][1], this_cam_data[1][2]], [other_cam_data[1][0], other_cam_data[1][1], other_cam_data[1][2]]])
		P1=np.array([[this_cam_data[0][0], this_cam_data[0][1], 0], [other_cam_data[0][0], other_cam_data[0][1], 0]])
		
		
		"""P0 and P1 are NxD arrays defining N lines.
		D is the dimension of the space. This function 
		returns the least squares intersection of the N
		lines from the system given by eq. 13 in 
		http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf.
		"""
		# generate all line direction vectors 
		n = (P1-P0)/np.linalg.norm(P1-P0,axis=1)[:,np.newaxis] # normalized

		# generate the array of all projectors 
		projs = np.eye(n.shape[1]) - n[:,:,np.newaxis]*n[:,np.newaxis]  # I - n*n.T

		# generate R matrix and q vector
		R = projs.sum(axis=0)
		q = (projs @ P0[:,:,np.newaxis]).sum(axis=0)

		# solve the least squares problem for the 
		# intersection point p: Rp = q
		p = np.linalg.lstsq(R,q,rcond=None)[0]
		
		print(f"LED at (%.2f, %.2f, %.2f)" % (round(p[0][0],2), round(p[1][0],2), round(p[2][0],2)) )
		return p
	except:
		print("Invalid data")
		return None

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
			break
			
socket_sv = Socket_Server(intersect)
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
socket_sv.join()
