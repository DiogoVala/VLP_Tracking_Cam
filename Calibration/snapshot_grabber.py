import cv2
import time
import sys
import os
import picamera
import numpy as np

# Settings
FRAME_WIDTH = 2016
FRAME_HEIGHT = 1520
SAVE_FOLDER = "./snapshots_"+str(FRAME_WIDTH)+"x"+str(FRAME_HEIGHT)
FILE_NAME = "snapshot"

def save_snaps(width, height, name, folder):

    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
            folder = os.path.dirname(folder)
            try:
                os.stat(folder)
            except:
                os.mkdir(folder)
    except:
        pass


    with picamera.PiCamera() as camera:
        print (f"Saving snapshots with resolution {width}x{height}.")
        print ("Press SPACE to capture.")
        camera.resolution=(width, height)
        frame = np.empty((height, width, 3), dtype=np.uint8) 
            
        nSnap=0
        fileName    = "%s/%s_%d_%d_" %(folder, name, width, height)
        while True:
            camera.capture(frame, 'rgb')
            
            frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame_resized = cv2.resize(frame_g, (int(width/2), int(height/2)))
            cv2.imshow('Snapshot Preview', frame_resized)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                print("Saving image ", nSnap)
                cv2.imwrite("%s%d.jpg"%(fileName, nSnap), frame_g)
                nSnap += 1

    cv2.destroyAllWindows()

def main():
    save_snaps(width=FRAME_WIDTH, height=FRAME_HEIGHT, name=FILE_NAME, folder=SAVE_FOLDER)

    print("Files saved.")

if __name__ == "__main__":
    main()
