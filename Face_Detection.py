import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video=cv2.VideoCapture(0)
jobs_image = face_recognition.load_image_file("jobs.jpg")

#if there are three faces in the frame the there will be three vectors
#face encoding is used to get a vector (numerical array) that represents
#the recongized face

jobs_encoding = face_recognition.face_encodings(jobs_image)[0]
ali_image= face_recognition.load_image_file("ali.jpg")
ali_encoding=face_recognition.face_encodings(ali_image)[0]

me_img= face_recognition.load_image_file("me.jpg")
me_encoding=face_recognition.face_encodings(me_img)[0]

face_inventory=[jobs_encoding,ali_encoding,me_encoding]
names=["Steve Jobs","Ali","Burhan"]

students=face_inventory.copy()
now=datetime.now()
#string format time which requires a format in which the date should
#be displayed
date_only=now.strftime("%Y-%m-%d")
#above input string is case sensitive
s=True
f=open(date_only+".csv",'w+',newline='')
#creating a write instance
lnwriter=csv.writer(f)

while True:
    #OpenCV reads a BGR frame by default 
    ret,frame=video.read()
    new_frame=cv2.resize(frame,(300,300))
    #converting to RGB because face_recognition expects RGB format
    rgb_new_frame=new_frame[:,:,::-1]
    if s:
        face_locations=face_recognition.face_locations(rgb_new_frame)
        face_encodings=face_recognition.face_encodings(rgb_new_frame,face_locations)

        for i in face_encodings:
            #it will return true or false array
            output=face_recognition.compare_faces(face_inventory,i)
            #it will return Euclidean distance (some accuracy)
            #this will be an array
            face_distance=face_recognition.face_distance(face_inventory,i)
            #used to get the index of minimum value in array
            best_fit=np.argmin(face_distance)
            #check the output array if the value at minimum index is 
            #True or not
            face_names=[]
            if output[best_fit]:
                #this is only recognizing the face not checking if it is 
                #present in inventory
                name=face_inventory[best_fit]
                face_names.append(name)

                #now checking if it is present in inventory or not
                #if name in face_inventory:
                    #cv2.putText(cv2.FONT_HERSHEY_SIMPLEX,(10,10),
                     #   1.5,(255,0,0),3,2)
                    #cv2.putText(img, text, coordinates, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)
    cv2.imshow("Attendence System",frame)
    if cv2.waitKey(1) == ord('b'):
        break

video.release()
cv2.destroyAllWindows()
f.close()
