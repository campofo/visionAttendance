import cv2
import face_recognition
#step 1
def face(real,test):
    img = face_recognition.load_image_file(real)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgTest = face_recognition.load_image_file(test)
    imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
    # step2
    faceLoc = face_recognition.face_locations(img)[0]
    encodeimg = face_recognition.face_encodings(img)[0]
    cv2.rectangle(img, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 4)

    faceLocTest = face_recognition.face_locations(imgTest)[0]
    encodeimgTest = face_recognition.face_encodings(imgTest)[0]
    cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 4)
    # step3
    results = face_recognition.compare_faces([encodeimg], encodeimgTest)
    faceDis = face_recognition.face_distance([encodeimg], encodeimgTest)
    print(faceDis)
    print(results)
    cv2.putText(imgTest, f'{results}{round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Pic", img)
    cv2.imshow("Pic tst", imgTest)
    cv2.waitKey(0)
face("Elon-Musk.jpg","images.jpeg")