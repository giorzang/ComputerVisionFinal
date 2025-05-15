import cv2
import os

def crop_center_image(input_path, output_path, crop_width=528, crop_height=960):
    img = cv2.imread(input_path)
    if img is None:
        print(f"Không thể đọc ảnh: {input_path}")
        return

    h, w = img.shape[:2]

    if w < crop_width or h < crop_height:
        print(f"Ảnh quá nhỏ: {input_path}")
        return

    # Tính toán tọa độ crop ở giữa
    x_start = (w - crop_width) // 2
    y_start = (h - crop_height) // 2

    cropped_img = img[y_start:y_start+crop_height, x_start:x_start+crop_width]
    cv2.imwrite(output_path, cropped_img)
    print(f"Đã lưu ảnh cắt giữa: {output_path}")

input_dir = "data/masks"        # Thư mục chứa ảnh gốc
output_dir = "data/cropped"      # Thư mục lưu ảnh đã cắt
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        crop_center_image(input_path, output_path)
