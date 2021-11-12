import numpy as np
import cv2
from PIL import Image
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import time

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret,frame = cam.read()
    #frame = cv2.flip(frame, 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters =  aruco.DetectorParameters_create()
    markers, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(frame.copy(), markers, ids)

    x_marker_center=[]
    y_marker_center=[]
    for marker in markers:
        for corner in marker:
            x_mean=int(np.mean(corner[:,0]))
            y_mean=int(np.mean(corner[:,1]))
            frame = cv2.circle(frame, (x_mean, y_mean), 1,(0,255,255), 20)
            x_marker_center.append(x_mean)
            y_marker_center.append(y_mean)

    if(len(x_marker_center)!=0):       
        img_center=(int(np.mean(x_marker_center)), int(np.mean(y_marker_center)))
        frame = cv2.circle(frame, img_center, 1,(0,255,255), 10)

    cv2.imshow('webcam',frame)

    if cv2.waitKey(1)&0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()