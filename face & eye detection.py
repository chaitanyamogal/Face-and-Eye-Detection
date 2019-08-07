import cv2,time
import numpy as np

face_haarcascades = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

video = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('video.avi',fourcc, 5.0,(640,480))

x = 1

while True :
      capture, frame = video.read()
      gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      faces = face_haarcascades.detectMultiScale(gray, 1.5, 5)
      for (x,y,w,h) in faces:
          cv2.rectangle(frame, (x,y), (x+w, y+h),(255,0,0),2)
          csm_gray = gray[y:y+h, x:x+w]
          csm_color =frame[y:y+h, x:x+w]
          eyes = eye_cascade.detectMultiScale(csm_gray)
          for (ex, ey, ew, eh) in eyes:
              cv2.rectangle(csm_color, (ex, ey), (ex+ew,ey+eh), (0,255,0),2)
          smile = smile_cascade.detectMultiScale(csm_gray)


      out.write(frame)
      cv2.imshow('capture',frame)
      key = cv2.waitKey(1)
      if key == ord('z'):
          break


time.sleep(3)
out.release()
video.release()
cv2.destroyAllWindows()