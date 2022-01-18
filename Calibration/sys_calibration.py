import numpy as np
import cv2
import re
from PIL import Image
import cv2.aruco as aruco
import picamera
import time
import math
from scipy.spatial.transform import Rotation

# CV2 Text settings
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.7
color = (0, 0, 255)
thickness = 1

# Camera Settings
RESOLUTION = (4032, 3040)
#RESOLUTION = (1280, 960)

# Camera matrix and distortion vector
calib_file = np.load("calib.npz")
mtx=calib_file['mtx']
dist=calib_file['dist']

# Real world position of corners
################  	  Corner 0 	 #####     Corner 1   #####     Corner 2 #######    Corner 3 #####
objp=np.array([[-115.5, -80.0, 0.0], [-79.5, -80.0, 0.0], [-79.5, -44.5, 0.0], [-115.5, -44.5,  0],\
			   [  79.5, -80.0, 0.0], [115.5, -80.0, 0.0], [115.5, -44.5, 0.0], [  79.5, -44.5,  0],\
			   [-115.5,  44.5, 0.0], [-79.5,  44.5, 0.0], [-79.5,  80.0, 0.0], [-115.5,  80.0,  0],\
			   [  79.5,  44.5, 0.0], [115.5,  44.5, 0.0], [115.5,  80.0, 0.0], [  79.5,  80.0,  0]],\
			   dtype = np.float32)

def flattenList(list):
	return [item for sublist in list for item in sublist]

def detectArucos(frame):
	aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
	parameters =  aruco.DetectorParameters_create()
	parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
	corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
	ret =  True if len(corners)!=0 else False

	return ret, corners, ids

def undistortFrame(frame):
	h, w = frame.shape[:2]
	# Obtain the new camera matrix and undistort the image
	newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
	undistortedFrame = cv2.undistort(frame, mtx, dist, None, newCameraMtx)
	# Crop the new frame to the ROI
	x, y, w, h = roi
	undistortedFrame = undistortedFrame[y:y + h, x:x + w]
	# Resize frame to original size
	undistortedFrame = cv2.resize(undistortedFrame, (RESOLUTION[0], RESOLUTION[1]), interpolation = cv2.INTER_LANCZOS4)

	return undistortedFrame

def drawCorner(frame, corner):
	x=int(corner[0])
	y=int(corner[1])
	frame = cv2.line(frame, (x-10, y), (x+10, y),(0,0,255), 2)
	frame = cv2.line(frame, (x, y-10), (x, y+10),(0,0,255), 2)
	# Draw the text
	cv2.putText(frame, str(i), (x,y), font, fontScale, color, thickness, cv2.LINE_AA)
	
def drawReprojection(frame, corner):
	x=int(corner[0])
	y=int(corner[1])
	frame = cv2.line(frame, (x-10, y), (x+10, y),(0,255,0), 2)
	frame = cv2.line(frame, (x, y-10), (x, y+10),(0,255,0), 2)

# main
with picamera.PiCamera() as camera:
	
	camera.resolution = RESOLUTION
	camera.exposure_mode = 'auto'
	frame_arr = np.empty((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)

	while True:
		tic=time.perf_counter()
		camera.capture(frame_arr, 'rgb')
		
		# Greyscale
		frame = cv2.cvtColor(frame_arr, cv2.COLOR_BGR2GRAY)
	
		# Undistort frame
		#frame = undistortFrame(frame)
		
		# Look for ArUco markers
		valid_markers, markers, ids = detectArucos(frame)
	
		if valid_markers:
			
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
					drawCorner(frame, corner)
					
				imgPts.append(corners) # Ordered list of corners
					
			
			# List of pixel coordinate for each found marker corner 
			imgPts=np.array(flattenList(imgPts), dtype = np.float32)
			#imgPts = cv2.cornerSubPix(frame,imgPts,(11,11),(-1,-1),criteria)
			
			# Get corresponding objp of the detected imgPts
			objpp=[]
			for id in sorted(ids):
				ii=id*4
				for i in range(4):
					objpp.append(list(objp[ii]))
					ii+=1
			objpp=np.array(objpp)
			
			
			# Find the rotation and translation vectors.
			ret, rvecs, tvecs = cv2.solvePnP(objpp, imgPts, mtx, dist)
			
			# Camera pose calculation 
			tvec = np.array(tvecs)
			rmat,_= cv2.Rodrigues(rvecs)
			rmat = np.array(rmat)
			rmat = rmat.T
			R = Rotation.from_matrix(rmat)
			
			camera_pos = -rmat @ tvec
			camera_ori= R.as_euler('xyz', degrees=True)
			
			print("Camera pos:\nx: %dmm\ny: %dmm\nz: %dmm" % (camera_pos[0], camera_pos[1], camera_pos[2]))
			print("Camera ori:\nx: %.2fº\ny: %.2fº\nz: %.2fº" % (camera_ori[0], camera_ori[1], camera_ori[2]))
			
			# Reproject objp in the image plane
			projs, jac = cv2.projectPoints(objpp, rvecs, tvecs, mtx, dist)
			
			'''
			for proj in projs:
				drawReprojection(frame, proj[0])
			'''
		toc = time.perf_counter()
		print(f"Calibration time: {toc - tic:0.4f} seconds")
		frame_resized = cv2.resize(frame, (int(RESOLUTION[0]/4), int(RESOLUTION[1]/4)))
		cv2.imshow('Picamera',frame_resized)
		
		
		key=cv2.waitKey(33)
		if key == ord('q'):
			break

