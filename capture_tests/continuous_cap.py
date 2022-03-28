from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2

resolution=(4032, 3040)
#resolution=(2720,2048)
#resolution=(2048,1536)
#resolution=(1024,768)
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = resolution
camera.framerate = 10
camera.exposure_mode = 'auto'
camera.iso 			 = 1600
rawCapture = PiRGBArray(camera, size=resolution)
#camera.start_preview()
#allow the camera to warmup
time.sleep(5)
print("Starting")

camera.start_recording('1.h264', bitrate=25000000, resize=(1920,1080))
camera.wait_recording(10)
camera.stop_recording()



'''
while True:
    
    for cap in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
        frame = cap.array
        frame1 = cv2.resize(frame, (int(resolution[0]/4), int(resolution[1]/4)))
        cv2.imshow('Picamera',frame1)
    
        rawCapture.truncate(0)
    
        key=cv2.waitKey(1)
        if key == ord('q'):
            break

'''

'''
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=False, burst=True):
    
    image = frame.array
    #image_resized = cv2.resize(image, (int(RESOLUTION[0]/4), int(RESOLUTION[1]/4)))
    cv2.imshow('Picamera',image)

    key=cv2.waitKey(1)
    if key == ord('q'):
        break

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

'''
