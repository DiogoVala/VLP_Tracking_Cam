import time
import threading 
import queue
import picamera
from picamera.array import PiYUVArray
import datetime
from PIL import Image
import cv2
import numpy as np
import math
from scipy.spatial.transform import Rotation
from numpy.linalg import inv
from sys_calibration_bare import *
import copy


# Run system calibration before starting camera
valid_markers, camera_pos, camera_ori, mapx, mapy, cameraMatrix, cameraDistortion, newCameraMatrix, rmat, tvec = runCalibration()
if(valid_markers == 0):
    print("Exiting program.")
    quit()

done = False # Global to indicate end of processing (To stop threads)
lock = threading.Lock() # 
pool = [] # Pool of ImageProcessor threads
queue = queue.Queue()

# Camera Settings and Startup
RESOLUTION = (4032, 3040)
rescale_factor = 16
crop_window = 100 

#RESOLUTION = (2016, 1520)
camera = picamera.PiCamera()
camera.resolution = RESOLUTION
camera.exposure_mode = 'auto'
camera.iso 	= 1600
camera.start_preview()
frame_T_ms=242 # Time between frames. Adjusted for frame acquisition delay
print("Camera warming up.")
time.sleep(1)

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

# Color detection thersholds (YUV)
lower_range = np.array([  0,  0, 83])
upper_range = np.array([233,255,255])

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


Nframes=0 # Number of processed frames
# Processing pipeline for each frame
def image_processor(frame):
    #print("Processing frame")
    '''
    global Nframes, done
    with lock:
        print("Acquired frames", Nframes)
        Nframes+=1
    
    if Nframes == 50:
        stop = datetime.datetime.now()
        elapsed = (stop - start).total_seconds()
        print("Framerate:",Nframes/elapsed)
        with lock:
            done=True
    '''
    print("")
    
    
    # Resize high resolution to low resolution
    tic = datetime.datetime.now()
    total = tic
    frame_low = cv2.resize(frame, (int(RESOLUTION[0]/rescale_factor),int(RESOLUTION[1]/rescale_factor)),interpolation = cv2.INTER_NEAREST) 
    print("cv2.resize (s):",(datetime.datetime.now()-tic).microseconds/1000000)
    
    # Filter by color to find LED
    tic = datetime.datetime.now()
    mask = cv2.inRange(frame_low, lower_range, upper_range)
    print("cv2.inRange (s):",(datetime.datetime.now()-tic).microseconds/1000000)
    
    # Blob detector
    tic = datetime.datetime.now()
    keypoints = detector_l.detect(mask)
    print("Detect Low(s):",(datetime.datetime.now()-tic).microseconds/1000000)
    
    if keypoints:
        coords = [keypoint.pt for keypoint in keypoints][0]
        coords = (coords[0]*rescale_factor, coords[1]*rescale_factor)
    
    x=int(coords[0])
    y=int(coords[1])

    frame_crop = frame[(y-crop_window):(y+crop_window), (x-crop_window):(x+crop_window)]
    
    # Filter by color to find LED on the new high resolution crop
    tic = datetime.datetime.now()
    mask = cv2.inRange(frame_crop, lower_range, upper_range)
    print("cv2.inRange (s):",(datetime.datetime.now()-tic).microseconds/1000000)
    
    # Blob detector
    tic = datetime.datetime.now()
    keypoints = detector_h.detect(mask)
    print("Detect High(s):",(datetime.datetime.now()-tic).microseconds/1000000)
		
    if keypoints:
        coords = [keypoint.pt for keypoint in keypoints][0]
        coords = (int(coords[0]+x-100), int(coords[1]+y-100))
    
    tic = datetime.datetime.now()
    undistorted_coords = cv2.undistortPoints(coords, cameraMatrix, cameraDistortion, None, newCameraMatrix)[0][0]
    print("cv2.undistortPoints (s):",(datetime.datetime.now()-tic).microseconds/1000000)
    
    tic = datetime.datetime.now()
    realWorld_coords = getWorldCoordsAtZ(undistorted_coords, 0, cameraMatrix, rmat, tvec)
    print("getWorldCoordsAtZ (s):",(datetime.datetime.now()-tic).microseconds/1000000)
    
    print(tuple(undistorted_coords), tuple([round(float(realWorld_coords[0]),2), round(float(realWorld_coords[1]),2)]))
    
    print("Total processing time (s):",(datetime.datetime.now()-total).microseconds/1000000)
    #queue.put(realWorld_coords)
    
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
                print(threading.current_thread())
                self.stream.seek(0)
                frame = self.stream.array
                #frame = np.asarray(Image.open(self.stream)) # Grab frame from stream
                self.processor_fcn(frame) # Call function to process frame
                self.terminated = True   
                '''
                try:
                    self.stream.seek(0)
                    frame = self.stream.array
                    #frame = np.asarray(Image.open(self.stream)) # Grab frame from stream
                    self.processor_fcn(frame) # Call function to process frame
                except:
                    print("Failed to acquire image from stream")
                    with lock:
                        done=True                        
                finally:
                    self.terminated = True   
                '''
                
# Generator of buffers for the capture_sequence method.
# Each buffer belongs to an ImageProcessor so each frame is sent to a different thread.
def streams(): 
    global prev_frame_time
    
    while not done:        
        with lock:
            frame_time = datetime.datetime.now() # Current time
            
            time_dif=frame_time-prev_frame_time # Time between previous frame and current time

            # Custom delay to ensure new frame is taken 250ms after previous frame
            delay_s=float(datetime.timedelta(milliseconds=frame_T_ms).microseconds - time_dif.microseconds)/1000000
            
            if delay_s < 0:
                delay_s = 0
            if delay_s > datetime.timedelta(milliseconds=frame_T_ms).microseconds/1000000:
                delay_s = datetime.timedelta(milliseconds=frame_T_ms).microseconds/1000000
            #print(delay_s)
            #time.sleep(delay_s)
            
            # This frame was taken at:
            prev_frame_time = datetime.datetime.now()
            
            processor = ImageProcessor(image_processor) # Create new thread to process new frame
            yield processor.stream
            processor.event.set()
    else:
        camera.close()


start = datetime.datetime.now()
prev_frame_time = datetime.datetime.now()
print("Starting capture.")
camera.capture_sequence(streams(), use_video_port=True, format='yuv')

print("Terminating program.")
