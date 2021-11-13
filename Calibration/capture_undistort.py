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

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
	
	ret,frame = cam.read()
	
	if frame is not None:
		
		#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		h, w = frame.shape[:2]
		# Obtain the new camera matrix and undistort the image
		newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
		undistortedFrame = cv2.undistort(frame, mtx, dist, None, newCameraMtx)
		
		# Crop the new frame to the ROI and resize to original size
		x, y, w, h = roi
		undistortedFrame = undistortedFrame[y:y + h, x:x + w]
		undistortedFrame = cv2.resize(undistortedFrame, (640,480), interpolation = cv2.INTER_LANCZOS4)

		frame = undistortedFrame
		
		# Aruco detection
		aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
		parameters =  aruco.DetectorParameters_create()
		markers, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
		frame_markers = aruco.drawDetectedMarkers(frame.copy(), markers, ids)

		x_marker_center=[]
		y_marker_center=[]
		for marker in markers:
			for corner in marker:
				x_mean=int(np.mean(corner[:,0]))
				y_mean=int(np.mean(corner[:,1]))
				frame = cv2.line(frame, (x_mean-10, y_mean), (x_mean+10, y_mean),(0,0,255), 2)
				frame = cv2.line(frame, (x_mean, y_mean-10), (x_mean, y_mean+10),(0,0,255), 2)
				x_marker_center.append(x_mean)
				y_marker_center.append(y_mean)

		if(len(x_marker_center)!=0):       
			img_center=(int(np.mean(x_marker_center)), int(np.mean(y_marker_center)))
			#frame = cv2.circle(frame, img_center, 1,(0,0,255), 3)
		
		cv2.imshow('Aruco detection with camera calibration',frame)
		
		
	
		#cv2.imshow('Calibration test', np.hstack((frame, undistortedFrame)))
	
	if cv2.waitKey(1)&0xFF == ord('q'):
		break

cam.release()
cv2.destroyAllWindows()
