import os
import cv2 as cv
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from arcface import ArcFace
import numpy as np
import face_searching

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

capture = cv.VideoCapture(0)
detector = MTCNN()
arcface = ArcFace()

if not capture.isOpened():
    print("❌ Không thể mở camera")
    exit()

frame_count = 1

while True:
    ret, frame = capture.read()
    if not ret:
        print("❌ Không thể đọc khung hình")
        break

    frame = cv.flip(frame, 1)  # Lật ngang giống gương
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Chuyển BGR → RGB
    results = detector.detect_faces(frame_rgb)
    if results:
        x, y, w, h = results[0]['box']
        cv.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Cắt ảnh khuôn mặt
        frame_count+=1
        face_img = frame[y: y+h, x: x+w]
        if face_img.shape[0] > 0 and face_img.shape[1] > 0 and frame_count % 10 == 0:
            face_img = cv.resize(face_img, (112, 112))
            face_img = np.expand_dims(face_img, axis=0)
            ypred = arcface.calc_emb(face_img)
            frame_count = 1
            predicted_name, confidence = face_searching.recognize_face_faiss(ypred)
            print(f"Đối tượng: {predicted_name}")
    frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)  # Chuyển lại về BGR trước khi hiển thị
    cv.imshow("Camera", frame_bgr)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
