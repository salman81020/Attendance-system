import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# automating to import the image from the directory

path = 'imagesAttendance'
images=[]
classNames = []
mylist = os.listdir(path)
for cls in mylist:
    curimg= cv2.imread(f'{path}/{cls}')
    images.append(curimg)
    classNames.append(os.path.splitext(cls)[0])

# creating the fuction to find the encodings for each one of image

def findencodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        encode =face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodeListKnown = findencodings(images)

# Marking attendance in comma separated value(CSV) file
def markAttendance(name):
     with open('Attendance.csv', 'r+') as f:
         mydatalist = f.readlines()
         namelist = []
         for line in mydatalist:
             entry = line.split(' , ')
             namelist.append(entry[0])
         if name not in namelist:
             now = datetime.now()
             dtstring = now.strftime("%H:%M:%S")
             f.writelines(f'\n{name},{dtstring}')

# capturing the img from video camera then turning it into encodes and comparing with our known face encodings

cap = cv2.VideoCapture(0)

while True:
    Success, img = cap.read()
    imgs = cv2.resize(img, (0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facecurframe = face_recognition.face_locations(imgs)
    encodecurframe = face_recognition.face_encodings(imgs, facecurframe)

    for encodeface,faceloc in zip(encodecurframe,facecurframe):
        matches = face_recognition.compare_faces(encodeListKnown, encodeface)
        facedis = face_recognition.face_distance(encodeListKnown, encodeface)
        print(facedis)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('webcam', img)
    cv2.waitKey(0)












