# Hướng Dẫn Cài Đặt Môi Trường Python

## 1. Cài Đặt Python
- Tải và cài đặt Python từ [python.org](https://www.python.org/downloads/)
- Đảm bảo thêm Python vào PATH trong quá trình cài đặt
- Kiểm tra phiên bản Python:
  ```sh
  python --version
  ```

## 2. Tạo Virtual Environment (Khuyến khích)
Sử dụng môi trường ảo để quản lý thư viện:
```sh
python -m venv venv
```
Kích hoạt môi trường ảo:
- **Windows (CMD/PowerShell):**
  ```sh
  venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```sh
  source venv/bin/activate
  ```

## 3. Cài Đặt Các Thư Viện Cần Thiết
Sau khi môi trường ảo được kích hoạt, chạy lệnh sau để cài đặt các thư viện từ `requirements.txt`:
```sh
pip install -r requirements.txt
```

## 4. Kiểm Tra Cài Đặt
Sau khi cài đặt xong, kiểm tra thư viện bằng cách chạy:
```sh
python -c "import numpy, cv2, tensorflow, faiss, mtcnn; print('Cài đặt thành công!')"
```
Nếu không có lỗi, bạn đã cài đặt thành công! 🎉

## 5. Cập Nhật `requirements.txt`
Nếu bạn cài thêm thư viện mới, hãy cập nhật `requirements.txt` bằng lệnh:
```sh
pip freeze > requirements.txt
```

## 6. Gỡ Cài Đặt Môi Trường
Nếu muốn xóa môi trường ảo:
```sh
rm -rf venv  # Mac/Linux
rd /s /q venv  # Windows
```

---
**Bây giờ bạn đã sẵn sàng chạy dự án! 🚀**

