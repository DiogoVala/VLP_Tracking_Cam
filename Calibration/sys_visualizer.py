import numpy as np
import cv2
import cv2.aruco as aruco
import picamera
from picamera.array import PiRGBArray
import time
import math
from scipy.spatial.transform import Rotation
from numpy.linalg import inv

# GPIO for light
import RPi.GPIO as GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(2, GPIO.OUT)

# CV2 Text settings
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 3
color = (0, 0, 255)
color1 = (255, 0, 0)
thickness = 3

# Camera Settings
RESOLUTION = (4032, 3040)
rescale_factor=8
crop_window = 80

# Blob detector (High Resolution)
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 70
params.maxArea = 6000
params.minDistBetweenBlobs = 50
params.filterByCircularity = True
params.minCircularity = 0.4
params.filterByConvexity = True
params.minConvexity = 0.2
params.filterByInertia = True
params.minInertiaRatio = 0.1
detector_h = cv2.SimpleBlobDetector_create(params)

# Blob detector (Rescaled Resolution)
params_low = cv2.SimpleBlobDetector_Params()
params_low.filterByArea = True
params_low.minArea = int(params.minArea/rescale_factor)
params_low.maxArea = int(params.maxArea*rescale_factor)
params_low.minDistBetweenBlobs = int(params.minDistBetweenBlobs/rescale_factor)
params_low.filterByCircularity = params.filterByCircularity
params_low.minCircularity = params.minCircularity
params_low.filterByConvexity = params.filterByConvexity
params_low.minConvexity = params.minConvexity
params_low.filterByInertia = params.filterByInertia
params_low.minInertiaRatio = params.minInertiaRatio
detector_l = cv2.SimpleBlobDetector_create(params_low)

# Color detection thersholds (YUV)
lower_range = np.array([  0,  0, 83])
upper_range = np.array([233,255,255])

# Camera matrix and distortion vector
calib_file = np.load("camera_intrinsics.npz")
mtx=calib_file['mtx']
dist=calib_file['dist']

# Real world position of corners
################  	  Corner 0 	 #####     Corner 1   #####     Corner 2 #######    Corner 3 #####
objp=np.array([[-115.5, -80.0, 0.0], [-79.5, -80.0, 0.0], [-79.5, -44.5, 0.0], [-115.5, -44.5,  0],\
			   [  79.5, -80.0, 0.0], [115.5, -80.0, 0.0], [115.5, -44.5, 0.0], [  79.5, -44.5,  0],\
			   [-115.5,  44.5, 0.0], [-79.5,  44.5, 0.0], [-79.5,  80.0, 0.0], [-115.5,  80.0,  0],\
			   [  79.5,  44.5, 0.0], [115.5,  44.5, 0.0], [115.5,  80.0, 0.0], [  79.5,  80.0,  0]],\
			   dtype = np.float32)

PoIs=[(2028, 1630), (2868, 1636), (1183, 1623), (2034, 1158) ,(2022, 2097)]

PoIs_proj=np.array([[0.0, 0.0, 0.0], [-79.5, 0.0, 0.0], [0.0, -44.5, 0.0], [0.0, 44.5,  0],\
					[79.5, 0.0, 0.0]],dtype = np.float32)

def flattenList(list):
	return [item for sublist in list for item in sublist]

def detectArucos(frame):
	tic = time.perf_counter()
	
	aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
	parameters =  aruco.DetectorParameters_create()
	parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
	corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
	ret =  True if len(corners)!=0 else False

	print(f"ArUco detection time: {time.perf_counter() - tic:0.4f} seconds")

	return ret, corners, ids

def undistortFrame(frame, mapx, mapy):
	tic = time.perf_counter()
	
	# Remaps the frame pixels to their new positions 
	frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
	
	print(f"Undistort time: {time.perf_counter() - tic:0.4f} seconds")
	return frame

def drawCorner(frame, corner, i):
	x=int(corner[0])
	y=int(corner[1])
	frame = cv2.line(frame, (x-30, y), (x+30, y),(0,0,255), 2)
	frame = cv2.line(frame, (x, y-30), (x, y+30),(0,0,255), 2)
	# Draw the text
	cv2.putText(frame, str(i), (x,y), font, fontScale, color, thickness, cv2.LINE_AA)
	
def drawReprojection(frame, corner):
	x=int(corner[0])
	y=int(corner[1])
	frame = cv2.line(frame, (x-30, y), (x+30, y),(0,255,0), 2)
	frame = cv2.line(frame, (x, y-30), (x, y+30),(0,255,0), 2)

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
    
def drawWorldAxis(frame):
	axis=np.array([[0.0, -100.0, 0.0], [0.0, 100.0, 0.0], [-135.0, 0.0, 0.0], [135.0, 0.0,  0] , [0.0, 0.0, 0.0], [0.0, 0.0, -400.0]],dtype = np.float32)
	projs, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, None)
	# Y axis
	frame = cv2.arrowedLine(frame, (int(round(projs[0][0][0],0)), int(round(projs[0][0][1], 0))), (int(round(projs[1][0][0],0)), int(round(projs[1][0][1],0))),(0,255,0), 5, tipLength = 0.05)
	# X axis
	frame = cv2.arrowedLine(frame, (int(round(projs[2][0][0],0)), int(round(projs[2][0][1], 0))), (int(round(projs[3][0][0],0)), int(round(projs[3][0][1],0))),(30,30,255), 5, tipLength = 0.05)

	#frame = cv2.line(frame, (int(round(projs[4][0][0],0)), int(round(projs[4][0][1], 0))), (int(round(projs[5][0][0],0)), int(round(projs[5][0][1],0))),(255,0,0), 2)

def drawRealWorld(x, y, frame):
	frame = cv2.line(frame, (x-30, y), (x+30, y),(255,0,0), 2)
	frame = cv2.line(frame, (x, y-30), (x, y+30),(255,0,0), 2)

	image_point=(x, y)
	world_coords=getWorldCoordsAtZ(image_point, 0.0, mtx, rmat, tvec)
	txt = f"({float(world_coords[0]):0.2f}, {float(world_coords[1]):0.2f})mm"
	cv2.putText(frame, txt, (x,y-10), font, fontScale/2, color1, thickness, cv2.LINE_AA)

def organizeObjpp(markers, ids):
	tic = time.perf_counter()
	
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
			drawCorner(frame, corner, i)
			
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
	
	print(f"Organize Objpp time: {time.perf_counter() - tic:0.4f} seconds")
	
	return objpp, imgPts
	

def detectBlob(frame):
	tic = time.perf_counter()

	frame_low = cv2.resize(frame, (int(RESOLUTION[0]/rescale_factor),int(RESOLUTION[1]/rescale_factor)),interpolation = cv2.INTER_NEAREST) 

	yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
	yuv_low = cv2.cvtColor(frame_low, cv2.COLOR_BGR2YUV)

	mask_low = cv2.inRange(yuv_low, lower_range, upper_range)

	# Blob detector
	keypoints_low = detector_l.detect(mask_low)

	leds_refined=[]

	if keypoints_low:
		leds_rough = [keypoint.pt for keypoint in keypoints_low]
		leds_rough = [(int(x)*rescale_factor, int(y)*rescale_factor) for x,y in leds_rough]

	for led in leds_rough:
		x=int(led[0])
		y=int(led[1])
		yuv_crop = yuv[(y-crop_window):(y+crop_window), (x-crop_window):(x+crop_window)]
		mask = cv2.inRange(yuv_crop, lower_range, upper_range)

		keypoints_high = detector_h.detect(mask)
		
		for keypoint_high in keypoints_high:
			led_refined = keypoint_high.pt
			led_refined = (round(led_refined[0])+x-crop_window, round(led_refined[1])+y-crop_window)
			leds_refined.append(led_refined)


	print(leds_refined)
	if leds_refined:
		for led in leds_refined:
			x=int(led[0])
			y=int(led[1])
			frame = cv2.line(frame, (x-30, y), (x+30, y),(0,0,255), 5)
			frame = cv2.line(frame, (x, y-30), (x, y+30),(0,0,255), 5)
	
	print(f"Blob detection time: {time.perf_counter() - tic:0.4f} seconds")
	return frame, mask_low
	

# main
with picamera.PiCamera() as camera:
	
	camera.resolution = RESOLUTION
	camera.exposure_mode = 'auto'
	camera.iso = 1600
	#camera.color_effects = (128, 128)
	capture = PiRGBArray(camera, size=RESOLUTION)
	h = RESOLUTION[1]
	w = RESOLUTION[0]
	
	# Get undistortion maps (This allows for a much faster undistortion using cv2.remap)
	newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
	mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newCameraMatrix, (w, h), cv2.CV_32FC1)

	while True:
		print("")
		
		tic=time.perf_counter()
		camera.capture(capture, 'rgb')
		
		frame = capture.array
		capture.truncate(0)
		
		# Undistort frame
		frame = undistortFrame(frame, mapx, mapy)
		
		yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
		
		# Detect LED blob
		frame, mask = detectBlob(frame)
		
		# Look for ArUco markers
		valid_markers, markers, ids = detectArucos(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
		
		if valid_markers:
			
			# Organize data obtained from detectedArucos
			objpp, imgPts = organizeObjpp(markers, ids)
			
			# Find the rotation and translation vectors.
			ret, rvecs, tvecs = cv2.solvePnP(objpp, imgPts, mtx, None)
			
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
			
			
			# Reproject ArUco corners
			projs, jac = cv2.projectPoints(objpp, rvecs, tvecs, mtx, None)
			for proj in projs:
				drawReprojection(frame, proj[0]) 
			
			'''
			# Reproject testing points
			projs1, jac1 = cv2.projectPoints(PoIs_proj, rvecs, tvecs, mtx, None)
			for proj in projs1:
				drawReprojection(frame, proj[0])
			for poi in projs1:
				poi=poi[0]
				drawRealWorld(int(round(poi[0],0)), int(round(poi[1],0)), frame)
			'''
			
			# Draw XY axis 
			drawWorldAxis(frame)
			
			'''
			# Draw information about camera pose
			txt = f"Camera position (xyz): {float(camera_pos[0]):0.2f}, {float(camera_pos[1]):0.2f} , {float(camera_pos[2]):0.2f} mm"
			cv2.putText(frame, txt, (700,100), font, fontScale, (255,255,255), thickness, cv2.LINE_AA)
			txt = f"Euler angles (xyz): {float(camera_ori[0]):0.2f}, {float(camera_ori[1]):0.2f} , {float(camera_ori[2]):0.2f} deg"
			cv2.putText(frame, txt, (700,200), font, fontScale, (255,255,255), thickness, cv2.LINE_AA)
			'''
		
		mask = cv2.inRange(yuv, lower_range, upper_range)
		cv2.imwrite("a.jpg", mask)
		
		toc = time.perf_counter()
		print(f"Calibration time: {toc - tic:0.4f} seconds")
		
		cv2.imshow('mask', cv2.resize(mask, (int(RESOLUTION[0]/4), int(RESOLUTION[1]/4))))
		cv2.imshow('Picamera',cv2.resize(frame, (int(RESOLUTION[0]/4), int(RESOLUTION[1]/4))))
		
		key=cv2.waitKey(33)
		if key == ord('q'):
			GPIO.output(2, GPIO.LOW)
			break
		if key == ord('l'): # Turn on the light
			GPIO.output(2, GPIO.HIGH)
		if key == ord('o'): # Turn off the light
			GPIO.output(2, GPIO.LOW)

