import os
import cv2
import numpy as np
from PIL import Image

path = "dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()
cascade_path = "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    raise IOError(f"Error: {cascade_path} not found")
detector = cv2.CascadeClassifier(cascade_path)


def getImagesAndLabel(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        try:
            PIL_img = Image.open(imagePath).convert("L")  # convert it to grayscale
            img_numpy = np.array(PIL_img, "uint8")
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(
                img_numpy, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            for x, y, w, h in faces:
                faceSamples.append(img_numpy[y : y + h, x : x + w])
                ids.append(id)
                # Tùy chọn: Hiển thị khuôn mặt được phát hiện
                cv2.rectangle(img_numpy, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.imshow("Face", img_numpy)
                cv2.waitKey(10)
        except Exception as e:
            print(f"Error processing file {imagePath}: {e}")
            continue
    cv2.destroyAllWindows()
    return faceSamples, ids


print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabel(path)

if faces and ids:
    recognizer.train(faces, np.array(ids))
    recognizer.write("trainer/trainer.yml")
    print(
        f"\n [INFO] Training Success. {len(np.unique(ids))} unique faces trained. Exiting ..."
    )
else:
    print("\n [INFO] No faces found to train. Exiting ...")
