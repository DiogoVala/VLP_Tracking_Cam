import numpy as np
import cv2
import re
from PIL import Image
import cv2.aruco as aruco
import picamera
import time
import math
from scipy.spatial.transform import Rotation
from numpy.linalg import inv

# Camera Settings
RESOLUTION = (4032, 3040)
camera = picamera.PiCamera()
camera.resolution 	 = RESOLUTION
camera.exposure_mode = 'auto'
camera.iso 			 = 1600

# Camera matrix and cameraDistortionortion vector
calib_file = np.load("camera_intrinsics.npz")
cameraMatrix=calib_file['mtx']
cameraDistortion=calib_file['dist']

#ArUco Settings
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
parameters =  aruco.DetectorParameters_create()
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

# Real world position of corners (in millimeters)
################  	  Corner 0 	 #####     Corner 1   #####     Corner 2 #######    Corner 3 #####
objp=np.array([[-115.5, -80.0, 0.0], [-79.5, -80.0, 0.0], [-79.5, -44.5, 0.0], [-115.5, -44.5,  0],\
			   [  79.5, -80.0, 0.0], [115.5, -80.0, 0.0], [115.5, -44.5, 0.0], [  79.5, -44.5,  0],\
			   [-115.5,  44.5, 0.0], [-79.5,  44.5, 0.0], [-79.5,  80.0, 0.0], [-115.5,  80.0,  0],\
			   [  79.5,  44.5, 0.0], [115.5,  44.5, 0.0], [115.5,  80.0, 0.0], [  79.5,  80.0,  0]],\
			   dtype = np.float32)

def flattenList(list):
	return [item for sublist in list for item in sublist]

def detectArucos(frame):
	corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

	return len(corners), corners, ids

def undistortFrame(frame, mapx, mapy):
	# Remaps the frame pixels to their new positions 
	frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
	return frame

def runCalibration():
	
	# Variable to store frame
	frame_new = np.empty((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
	h, w = frame_new.shape[:2]

	# Get undistortion maps (This allows for a much faster undistortion using cv2.remap)
	newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, cameraDistortion, (w, h), 1, (w, h))
	mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, cameraDistortion, None, newCameraMatrix, (w, h), cv2.CV_32FC1)

	# Capture frame from PiCamera
	camera.capture(frame_new, 'rgb')
	
	# ArUco detection is faster in grayscale
	frame = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)
	
	# Undistort frame
	frame = undistortFrame(frame, mapx, mapy)

	# Look for ArUco markers
	valid_markers, markers, ids = detectArucos(frame)
	
	if valid_markers > 0: # Number of detected ArUco Markers
		
		# Flatten lists to make them more maneageable 
		markers = flattenList(markers)
		ids = flattenList(ids)

		# Here we create a list of found marker corners.
		imgPts=[]
		for m_id in sorted(ids):
			
			idx = ids.index(m_id)
			marker = markers[idx]
			
			corners=[]
			for i, corner in enumerate(marker):
				x=int(corner[0])
				y=int(corner[1])
				corners.append([x, y])
				
			imgPts.append(corners) # Ordered list of corners
				
		# List of pixel coordinate for each found marker corner 
		imgPts=np.array(flattenList(imgPts), dtype = np.float32)

		# Get corresponding objp of the detected imgPts
		objpp=[]
		for id in sorted(ids):
			ii=id*4
			for i in range(4):
				objpp.append(list(objp[ii]))
				ii+=1
		objpp=np.array(objpp)
		
		# Find the rotation and translation vectors.
		ret, rvecs, tvecs = cv2.solvePnP(objpp, imgPts, cameraMatrix, None)
		
		# Camera pose calculation 
		tvec = np.array(tvecs)
		rmat,_= cv2.Rodrigues(rvecs)
		rmat = np.array(rmat)
		rmat = rmat.T
		R = Rotation.from_matrix(rmat)
		
		camera_pos = -rmat @ tvec
		camera_ori= R.as_euler('xyz', degrees=True)
		
		#print("Camera pos:\nx: %dmm\ny: %dmm\nz: %dmm" % (camera_pos[0], camera_pos[1], camera_pos[2]))
		#print("Camera ori:\nx: %.2fº\ny: %.2fº\nz: %.2fº" % (camera_ori[0], camera_ori[1], camera_ori[2]))
		
		print("Calibration Complete.")
		print("Number of markers detected:", valid_markers)
	return valid_markers, camera_pos, camera_ori, mapx, mapy 
