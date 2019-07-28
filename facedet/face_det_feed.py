import cv2
import sys
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

while True:
	ret, frame = video_capture.read()
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	faces = face_cascade.detectMultiScale( gray, 1.1, 5, minSize = (5,5))
	
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		
		
	cv2.imshow('video', frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
video_capture.release()
cv2.destroyAllWindows()