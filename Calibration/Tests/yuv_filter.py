
#finding hsv range of target object(pen)
import cv2
import numpy as np
import time
from PIL import Image
import picamera
from picamera.array import PiYUVArray
# GPIO for light
import RPi.GPIO as GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(2, GPIO.OUT)

# A required callback method that goes into the trackbar function.
def nothing(x):
    pass

# Initializing the webcam feed.
RESOLUTION = (4032, 3040)
camera=picamera.PiCamera()
camera.resolution = RESOLUTION
camera.exposure_mode = 'auto'
camera.iso = 1600
capture = PiYUVArray(camera, size=RESOLUTION)

lower_range = np.array([0,0,76])
upper_range = np.array([203,255,207])
# Create a window named trackbars.
cv2.namedWindow("Trackbars")

# Now create 6 trackbars that will control the lower and upper range of 
# H,S and V channels. The Arguments are like this: Name of trackbar, 
# window name, range,callback function. For Hue the range is 0-179 and
# for S,V its 0-255.
cv2.createTrackbar("L - Y", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - U", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 76, 255, nothing)
cv2.createTrackbar("U - Y", "Trackbars", 203, 255, nothing)
cv2.createTrackbar("U - U", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 207, 255, nothing)

while True:

    # Start reading the webcam feed frame by frame.
    camera.capture(capture, 'yuv')
    frame = capture.array
    capture.truncate(0)
     
    # Get the new values of the trackbar in real time as the user changes 
    # them
    l_y = cv2.getTrackbarPos("L - Y", "Trackbars")
    l_u = cv2.getTrackbarPos("L - U", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_y = cv2.getTrackbarPos("U - Y", "Trackbars")
    u_u = cv2.getTrackbarPos("U - U", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    thearray = [[l_y,l_u,l_v],[u_y, u_u, u_v]]
    print(thearray)
    # Set the lower and upper HSV range according to the value selected
    # by the trackbar
    lower_range = np.array([l_y, l_u, l_v])
    upper_range = np.array([u_y, u_u, u_v])
    
    # Filter the image and get the binary mask, where white represents 
    # your target color
    mask = cv2.inRange(frame, lower_range, upper_range)
    
    # You can also visualize the real part of the target color (Optional)
    #res = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Converting the binary mask to 3 channel image, this is just so 
    # we can stack it with the others
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # stack the mask, orginal frame and the filtered result
    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
    stacked = np.hstack((mask,frame))
    
    # Show this stacked frame at 40% of the size.
    cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.15,fy=0.15))
    
    # If the user presses ESC then exit the program
    key=cv2.waitKey(33)
    if key == ord('q'):
        GPIO.output(2, GPIO.LOW)
        break
    if key == ord('l'):
        GPIO.output(2, GPIO.HIGH)
    if key == ord('o'):
        GPIO.output(2, GPIO.LOW)
    
    # If the user presses `s` then print this array.
    if key == ord('s'):
        
        thearray = [[l_y,l_u,l_v],[u_y, u_u, u_v]]
        print(thearray)
        
        # Also save this array as penval.npy
        np.save('hsv_value',thearray)
        break
    
# Release the camera & destroy the windows.    
camera.close()
cv2.destroyAllWindows()
