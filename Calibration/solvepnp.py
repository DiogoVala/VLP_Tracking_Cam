import numpy as np
import cv2
import re
from PIL import Image
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import time

calib_file = np.load("calib.npz")
mtx=calib_file['mtx']
dist=calib_file['dist']

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
objp = [[207, 0, 0], []

print(objp)

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def detectArucos(frame):
	aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
	parameters =  aruco.DetectorParameters_create()
	corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
	#frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
	ret=  True if len(corners)!=0 else False
	return ret, corners, ids
	

def undistortFrame(frame):
	h, w = frame.shape[:2]
	# Obtain the new camera matrix and undistort the image
	newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
	undistortedFrame = cv2.undistort(frame, mtx, dist, None, newCameraMtx)
	# Crop the new frame to the ROI and resize to original size
	x, y, w, h = roi
	undistortedFrame = undistortedFrame[y:y + h, x:x + w]
	undistortedFrame = cv2.resize(undistortedFrame, (640,480), interpolation = cv2.INTER_LANCZOS4)

	return undistortedFrame
	

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


while True:
	
	ret,frame = cam.read()
	
	if ret == True:
		
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		frame = undistortFrame(frame)
			
		#ret, corners = cv2.findChessboardCorners(frame, (9,6), None)
		#print(corners)
		
		# Aruco detection
		ret, corners, ids = detectArucos(frame)
		
		if ret == True:
			
			imgPts=[]
			for corner in corners:
				imgPts.append(corner)
			'''
			imgPts=np.array(imgPts, dtype='f')
			imgPtsSub = cv2.cornerSubPix(frame,imgPts,(11,11),(-1,-1),criteria)
			print(imgPts, imgPtsSub)
			'''
		'''	
		if ret == True:
			#corners2 = cv2.cornerSubPix(frame,corners,(11,11),(-1,-1),criteria)
			# Find the rotation and translation vectors.
			ret,rvecs, tvecs = cv2.solvePnP(objp, imgPts, mtx, dist)
			# project 3D points to image plane
			imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
			frame = draw(frame,corners2,imgpts)
			k = cv2.waitKey(0) & 0xFF
			if k == ord('s'):
				cv2.imwrite(fname[:6]+'.png', img)
		'''
		'''
		reference_pts=[]
		for marker in markers:
			for corner in marker:
				drawCorner(frame, corner)
				for vertex in corner:
					reference_pts.append(tuple(vertex))

		'''
		cv2.imshow('solvepnp',frame)
		#cv2.imshow('Aruco detection with camera calibration',frame_markers)
		
	
		#cv2.imshow('Calibration test', np.hstack((frame, undistortedFrame)))
	
	if cv2.waitKey(1)&0xFF == ord('q'):
		break

cam.release()
cv2.destroyAllWindows()
