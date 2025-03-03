import os
import cv2 as cv
import numpy as np
import pandas as pd
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet

# Ẩn cảnh báo TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Khởi tạo model
detector = MTCNN()
facenet = FaceNet()

# Danh sách để lưu dữ liệu
data = []

folder_path = "dataset"

for root, dirs, files in os.walk(folder_path):
    label = os.path.basename(root)  # Lấy tên thư mục làm nhãn
    print(f"📂 Đọc thư mục: {label}")

    for file in files:
        file_path = os.path.join(root, file)
        print(f"  📄 Xử lý: {file_path}")

        # Đọc ảnh
        img_bgr = cv.imread(file_path)
        if img_bgr is None:
            print(f"⚠️ Lỗi đọc ảnh: {file_path}")
            continue

        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        results = detector.detect_faces(img_rgb)

        if results:
            x, y, w, h = results[0]['box']

            # Cắt ảnh khuôn mặt
            face_img = img_rgb[y:y+h, x:x+w]
            if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                face_img = cv.resize(face_img, (160, 160))
                face_img = np.expand_dims(face_img, axis=0)

                # Trích xuất embedding
                ypred = facenet.embeddings(face_img)

                # Lưu vào danh sách với nhãn (tên thư mục)
                data.append([label] + ypred.flatten().tolist())

# Chuyển danh sách thành DataFrame
df = pd.DataFrame(data)

# Đặt tên cột
df.columns = ["label"] + [f"dim_{i}" for i in range(df.shape[1] - 1)]

# Lưu vào file CSV
df.to_csv("face_embeddings.csv", index=False)

print("✅ Đã lưu face_embeddings.csv thành công!")


import faiss

# ==== Bước 1: Chuẩn bị dữ liệu (vector từ FaceNet) ====
face_db = {
    "Alice": [np.random.rand(512) for _ in range(5)],
    "Bob": [np.random.rand(512) for _ in range(5)],
    "Charlie": [np.random.rand(512) for _ in range(5)],
}

# Tạo danh sách nhãn (labels) và ma trận vector
labels = []
vectors = []
index_to_name = {}  # Ánh xạ chỉ số FAISS -> tên người

i = 0
for name, vecs in face_db.items():
    for v in vecs:
        labels.append(i)
        vectors.append(v)
        index_to_name[i] = name
        i += 1

# Chuyển dữ liệu thành numpy array
vectors = np.array(vectors).astype('float32')  # FAISS yêu cầu float32

# ==== Bước 2: Khởi tạo FAISS Index ====
dimension = 512  # Vector FaceNet có 512 chiều
index = faiss.IndexFlatL2(dimension)  # Sử dụng L2 (Euclidean Distance)
index.add(vectors)  # Thêm tất cả vector vào FAISS

# ==== Bước 3: Tìm kiếm vector mới ====
def recognize_face_faiss(face_vector, top_k=1, threshold=1.0):
    """
    Tìm người gần nhất với face_vector bằng FAISS.
    Nếu khoảng cách > threshold, trả về 'Unknown'.
    """
    face_vector = np.array(face_vector).astype('float32').reshape(1, -1)  # Định dạng lại vector
    D, I = index.search(face_vector, top_k)  # D: khoảng cách, I: chỉ số

    best_index = I[0][0]  # Lấy chỉ số gần nhất
    best_distance = D[0][0]

    if best_distance > threshold:  # Nếu khoảng cách quá xa, coi là "Unknown"
        return "Unknown", best_distance

    return index_to_name[best_index], best_distance

# ==== Bước 4: Test nhận diện ====
new_face = np.random.rand(512)  # Vector từ FaceNet
predicted_name, confidence = recognize_face_faiss(new_face)

print(f"Predicted: {predicted_name} (Distance: {confidence:.2f})")
