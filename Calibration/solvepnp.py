import numpy as np
import cv2
import re
from PIL import Image
from cv2 import aruco
import time

# GPIO for light
import RPi.GPIO as GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(2, GPIO.OUT)

# CV2 Text settings
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.7
color = (0, 0, 255)
thickness = 1

# Camera matrix and distortion vector
calib_file = np.load("calib.npz")
mtx=calib_file['mtx']
dist=calib_file['dist']

# Video capture settings
frame_width=640
frame_height=480
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Real world position of corners
################  Corner 0 ##### Corner 1 ##### Corner 2 ####### Corner 3 #########
objp=np.array([[122,  85, 0], [122,  47, 0], [ 85,   47, 0], [  85,  85, 0],\
			   [122, -47, 0], [122, -85, 0], [ 85,  -85, 0], [  85, -47, 0],\
			   [-85, -47, 0], [-85, -85, 0], [-122, -85, 0], [-122, -47, 0],\
			   [-85,  85, 0], [-85,  47, 0], [-122,  47, 0], [-122,  85, 0]],\
			   dtype = np.float32)

print(objp)
def flattenList(list):
	return [item for sublist in list for item in sublist]

def detectArucos(frame):
	aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
	parameters =  aruco.DetectorParameters_create()
	corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
	ret=  True if len(corners)!=0 else False
	
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
	undistortedFrame = cv2.resize(undistortedFrame, (frame_width,frame_height), interpolation = cv2.INTER_LANCZOS4)

	return undistortedFrame
	
def drawCorner(frame, corner):
	x=int(corner[0])
	y=int(corner[1])
	frame = cv2.line(frame, (x-10, y), (x+10, y),(0,0,255), 2)
	frame = cv2.line(frame, (x, y-10), (x, y+10),(0,0,255), 2)
	# Draw the text
	cv2.putText(frame, str(i), (x,y), font, fontScale, color, thickness, cv2.LINE_AA)

while True:
	
	ret,frame = cam.read()
	
	if frame is not None:
		
		#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		frame = undistortFrame(frame)
		
		ret, markers, ids = detectArucos(frame)

		if(ret == True):
			#imgPts = [[[0, 0] for x in range(4)] for y in range(4)] 
			imgPts=[]
			
			markers = flattenList(markers)
			ids = flattenList(ids)

			for m_id in sorted(ids):
				
				idx = ids.index(m_id)
				marker = markers[idx]

				corners=[]
				for i, corner in enumerate(marker):
					x=int(corner[0])
					y=int(corner[1])
					corners.append([x, y])
					#imgPts[m_id][i]=[x, y]
					drawCorner(frame, corner)
				imgPts.append(corners)
					
			#print(flattenList(imgPts))
			imgPts=np.array(flattenList(imgPts), dtype = np.float32)
			print(imgPts)
			
			# Get corresponding objp of the detected imgPts
			objpp=[]
			for id in sorted(ids):
				ii=id*4;
				for i in range(4):
					objpp.append(list(objp[ii]))
					ii+=1
			objpp=np.array(objpp)
			
			# Find the rotation and translation vectors.
			ret, rvecs, tvecs = cv2.solvePnP(objpp, imgPts, mtx, dist)
			print("Rotation:", rvecs)
			print("Translation:", tvecs)
			# Project 3D points to image plane
			#imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

		cv2.imshow('Aruco detection with camera calibration',frame)
		#cv2.imshow('Aruco detection with camera calibration',frame_markers)
		
	key=cv2.waitKey(33)
	if key == ord('q'):
		GPIO.output(2, GPIO.LOW)
		GPIO.cleanup()
		break
	if key == ord('l'):
		GPIO.output(2, GPIO.HIGH)
	if key == ord('o'):
		GPIO.output(2, GPIO.LOW)
		#cv2.imshow('Aruco detection with camera calibration',frame_markers)
		
	
		#cv2.imshow('Calibration test', np.hstack((frame, undistortedFrame)))
	
	if cv2.waitKey(1)&0xFF == ord('q'):
		break

cam.release()
cv2.destroyAllWindows()
