import numpy as np
import cv2
import re
from PIL import Image
import cv2.aruco as aruco
import picamera
import time
import math
from scipy.spatial.transform import Rotation

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


with picamera.PiCamera() as camera:
    
    camera.resolution = RESOLUTION
    frame_arr = np.empty((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)

    while True:
        camera.capture(frame_arr, 'rgb')
        
        frame = cv2.cvtColor(frame_arr, cv2.COLOR_BGR2GRAY)
    
        frame = undistortFrame(frame)
        


        cv2.imshow('Picamera',frame)
        key=cv2.waitKey(33)
        if key == ord('q'):
            break

