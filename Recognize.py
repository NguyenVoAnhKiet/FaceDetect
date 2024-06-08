import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = ["KHANH", "KIET", "BAO", "3", "4", "5"]

cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

confidence_threshold = 75  # Adjust this threshold as needed

while True:
    ret, img = cam.read()

    if not ret:
        print("Error: Failed to capture image")
        break

    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y : y + h, x : x + w])

        if confidence < confidence_threshold and id<=len(names)-1:
            id = names[id]
            confidence_percent = 100 - confidence
            label = f"{id} ({confidence_percent:.2f}%)"
        else:
            label = "Unknown"

        cv2.putText(img, label, (x + 5, y - 5), font, 1, (0, 255, 0), 2)
        # cv2.putText(img, label, (x + 5, y + h - 5), font, 1, (0, 255, 0), 2)

    cv2.imshow("Face Detected", img)
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
