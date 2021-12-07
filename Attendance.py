import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
path = "imagesAll"
images =[]
className =[]
myList = os.listdir(path)
#print(myList)
for list in myList:
    listImg = cv2.imread(f'{path}/{list}')
    images.append(listImg)
    className.append(os.path.splitext(list)[0])
#print(className)
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
encodeList = findEncodings(images)
print("ENCODING COMPLETE")

def markAttendance(name):
    with open("Attendance.csv","r+") as f:
        myDataList =f.readlines()
        print(myDataList)
        nameList=[]
        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            ttString = now.strftime("%H:%M:%S")
            dtString = now.strftime('%d/%m/%Y')

            f.writelines(f'\n{name},{ttString},{dtString}')

cap = cv2.VideoCapture(0)

while True:
    frames,img =cap.read()
    imgS =cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceLocCur = face_recognition.face_locations(imgS)
    encodesCur = face_recognition.face_encodings(imgS,faceLocCur)

    for encodeFac,facLoc in zip(encodesCur,faceLocCur):
        matches = face_recognition.compare_faces(encodeList,encodeFac)
        faceDis = face_recognition.face_distance(encodeList,encodeFac)
        #print(faceDis)
        #cv2.imshow("man",img)
        #cv2.waitKey(0)
        matchIndex = np.argmin(faceDis)
        #print(matchIndex)

        if matches[matchIndex]:
            name=className[matchIndex].upper()
            #print(name)
            cv2.putText(img,f'{name} RECORDED',(70,70),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            markAttendance(name)
    cv2.imshow("Face ",img)
    cv2.waitKey(1)






