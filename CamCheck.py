import cv2

cam = cv2.VideoCapture(0)
while True:
    ret, img = cam.read()
    if not ret:
        print("Error: Failed to capture image")
        break
    cv2.imshow("Camera", img)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
