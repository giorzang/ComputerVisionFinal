#### Cài đặt các thư viện cần thiết
```bash
pip install tensorflow==2.10.0 tensorflow-gpu==2.10.0
pip install opencv-python numpy<2 matplotlib scikit-learn Pillow
```
- `tensorflow`: Framework học sâu chính.
- `opencv-python`(`cv2`): Để đọc, ghi và xử lý ảnh.
- `numpy`: Để làm việc với mảng số hiệu quả.
- `matplotlib`: Để vẽ đồ thị và hiển thị ảnh.
- `scikit-learn`: Để chia tập dữ liệu train/validation.
- `Pillow` (`PIL`): Thư viện xử lý ảnh, đôi khi hữu ích cho việc kiểm tra định dạng.

#### Giải thích `src/data_loader.py`:
