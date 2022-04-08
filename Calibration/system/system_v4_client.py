import io
import time
import threading
import picamera
import datetime
import math
import cv2
import numpy as np
from numpy.linalg import inv
from numpy import array, cross
from numpy.linalg import solve, norm
from scipy.spatial.transform import Rotation
from sys_calibration_bare import *
from sys_connection import *
from image_processor import *

# Camera Settings
camera_resolution = (4032, 3040)
rescale_factor = 16
crop_window = 100

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
			try:
				mask = cv2.inRange(yuv_crop, lower_range, upper_range)
			except:
				break

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
		
	# Send location data to the server
	socket_clt.txdata=this_cam_data
	socket_clt.event.set()
	#print("Camera coordinates:", camera_pos)

	#print("Total processing time (s):",(datetime.datetime.now()-total).microseconds/1000000)
	#queue.put(realWorld_coords)

	return

####### MAIN ####### 
print("Starting client camera.")

# Initialize Socket Server
socket_clt = Socket_Client()

# Run system calibration before starting camera (Must be done before creating a PiCamera instance)
valid_markers, camera_pos, camera_ori, mapx, mapy, cameraMatrix, cameraDistortion, newCameraMatrix, rmat, tvec = runCalibration()
if(valid_markers == 0):
	print("Exiting program.")
	quit()

# Camera startup
camera = picamera.PiCamera()
camera.resolution = camera_resolution
camera.exposure_mode = 'auto'
camera.iso 	= 1600
print("Camera warming up.")
time.sleep(1)

# Initialize pool of threads to process each frame
global ImgProcessorPool
ImgProcessorPool = [ImageProcessor(image_processor, camera, camera_resolution) for i in range(nProcess)]

print("Starting capture.")
camera.capture_sequence(getStream(), use_video_port=True, format='yuv')

while pool:
	with ImgProcessorLock:
		processor = pool.pop()
	processor.terminated = True
	processor.join()
socket_clt.join()
print("Terminating program.")


print("Terminating program.")
while pool:
	with lock:
		processor = pool.pop()
	processor.terminated = True
	processor.join()
socket_clt.join()
