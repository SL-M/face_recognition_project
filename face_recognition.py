import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('har_face.xml')

people = ['Brad Pitt', 'Edward Norton', 'Matt LeBlanc', 'matthew mcconaughey', 'Robert de niro']
# fearures = np.load('features.npy')
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_traind.yml')

img = cv.imread('Faces/Test/Matt LeBlanc/in-2004-matt-leblanc-became-a-father-1700491669.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect the face
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]}, confidence = {confidence}')

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255.0),thickness=2)
    cv.rectangle(img,(x,y), (x+w,y+h), (0,255,0), thickness=2)
cv.imshow('Detected face', img)

cv.waitKey(0)