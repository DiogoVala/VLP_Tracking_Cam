import cv2
import numpy as np
import datetime

RESOLUTION = (4032, 3040)
rescale_factor=8

frame = cv2.imread("a.jpg")

frame_low = cv2.resize(frame, (int(RESOLUTION[0]/rescale_factor),int(RESOLUTION[1]/rescale_factor)),interpolation = cv2.INTER_NEAREST) 

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

# Resize high resolution to low resolution
tic = datetime.datetime.now()
frame_low = cv2.resize(frame, (int(RESOLUTION[0]/rescale_factor),int(RESOLUTION[1]/rescale_factor)),interpolation = cv2.INTER_NEAREST) 
print("cv2.resize (s):",(datetime.datetime.now()-tic).microseconds/1000000)

mask = frame_low

# Blob detector
tic = datetime.datetime.now()
keypoints = detector_l.detect(mask)
print("Detect Low(s):",(datetime.datetime.now()-tic).microseconds/1000000)

if keypoints:
	coords = [keypoint.pt for keypoint in keypoints][0]
	coords = (coords[0]*rescale_factor, coords[1]*rescale_factor)

x=int(coords[0])
y=int(coords[1])

frame_crop = frame[(y-100):(y+100), (x-100):(x+100)]
cv2.imshow("crop", frame_crop)

mask = frame_crop

# Blob detector
tic = datetime.datetime.now()
keypoints = detector_l.detect(mask)
print("Detect High(s):",(datetime.datetime.now()-tic).microseconds/1000000)
	
if keypoints:
	coords = [keypoint.pt for keypoint in keypoints][0]
	coords = (int(coords[0]+x-100), int(coords[1]+y-100))

print(coords)

x=int(coords[0])
y=int(coords[1])

frame = cv2.line(frame, (x-30, y), (x+30, y),(0,0,255), 2)
frame = cv2.line(frame, (x, y-30), (x, y+30),(0,0,255), 2)

cv2.imshow("Keypoints", cv2.resize(frame,None,fx=0.3,fy=0.3))

cv2.waitKey(0)
