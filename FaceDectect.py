import cv2
import os
import numpy as np

# from tensorflow import ImageDataGenerator
from keras.src.legacy.preprocessing.image import ImageDataGenerator


# Function for augmenting images
def augment_image(image):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Add channel dimension for grayscale image
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    it = datagen.flow(image, batch_size=1)
    batch = next(it)
    image_aug = batch[0].astype("uint8")

    # Ensure the shape is correct before squeezing
    if image_aug.shape[0] == 1:
        image_aug = np.squeeze(image_aug, axis=0)  # Remove batch dimension
    if image_aug.shape[-1] == 1:
        image_aug = np.squeeze(image_aug, axis=-1)  # Remove channel dimension
    return image_aug


# Kiểm tra và tạo thư mục dataset
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Mở camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open video device.")
    exit(1)

cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# Tải mô hình Haar Cascade
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if face_detector.empty():
    print("Error loading haarcascade_frontalface_default.xml")
    exit(1)

# Nhập ID người dùng
face_id = input("\n enter user id end press <return> ==> ")

print("\n [INFO] Khởi tạo camera...")
count = 0

# Vòng lặp chính
while True:
    ret, img = cam.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        face_img = gray[y : y + h, x : x + w]
        face_path = f"dataset/User.{face_id}.{count}.jpg"
        cv2.imwrite(face_path, face_img)

        # Generate augmented images
        for i in range(5):  # Generate 5 augmented images per original image
            augmented_img = augment_image(face_img)
            augmented_path = f"dataset/User.{face_id}.{count}.{i}.jpg"
            cv2.imwrite(augmented_path, augmented_img)

        cv2.imshow("image", img)

    k = cv2.waitKey(100) & 0xFF
    if k == 27:  # ESC để thoát
        break
    elif count >= 50:  # Chụp 0 ảnh khuôn mặt
        break

print("\n [INFO] Exiting...")
cam.release()
cv2.destroyAllWindows()
