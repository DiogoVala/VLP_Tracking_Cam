
from fps import FPS
from stream import PiStream
import cv2
import time

vs = PiStream(src=0).start()
time.sleep(2.0)
resolution=(4032, 3040)
i=0

while True:
	frame = vs.read()
	frame1 = cv2.resize(frame, (int(resolution[0]/4), int(resolution[1]/4)))
	cv2.imshow("Frame",frame1)
	print(i)
	i=i+1
	key=cv2.waitKey(1)
	if key == ord('q'):
	    break

cv2.destroyAllWindows()
vs.stop()
