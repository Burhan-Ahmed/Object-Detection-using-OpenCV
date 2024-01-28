import pandas as pd
import numpy as np 
import face_recognition
import cv2
import os
import csv
from datetime import datetime

#creating an instace of the img to be used further
img=cv2.imread("212.jpg",-1)
#img=cv2.resize(img,(500,500))

cv2.imshow('PIC',img)

#will be closed itself after 5 seconds
cv2.waitKey(1000)
cv2.destroyAllWindows()

#this will show the numpy array of the pixels that represent this image
print(img)

print(img.shape)
# (393,300,3)
# (rows, cols, channels) channels represents the color combination for each pixel

# use VideoCapture(0) for camera where 0 represents the camera number and we can also use
#video here by replacing the clip name instead of 0

pic=cv2.VideoCapture(0)

while True:
    ret,frame=pic.read()
    width=int(pic.get(3))
    height=int(pic.get(4))

    # (picture, (x,y),(R,G,B))
    cv2.line(frame,(0,0),(width,height),(255,0,0))
    cv2.line(frame,(0,height),(width,0),(255,0,0))


    cv2.imshow("My Video", frame)

    if cv2.waitKey(1)==ord('b'):
        break

pic.release()
cv2.destroyAllWindows()