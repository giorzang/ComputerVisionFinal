# Thay nền thời gian thực sử dụng mô hình DeepLabV3 với độ phân giải cao 540x960
import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
import time

def smart_resize(img, target_size=(256, 144)):
    h, w = img.shape[:2]
    target_w, target_h = target_size
    
    # Tính tỉ lệ scale giữ nguyên aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize ảnh gốc
    resized = cv2.resize(img, (new_w, new_h))
    
    # Tạo ảnh nền đen với kích thước đích
    new_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Tính toán vị trí đặt ảnh đã resize
    x_start = (target_w - new_w) // 2
    y_start = (target_h - new_h) // 2
    
    # Đặt ảnh vào giữa nền đen
    new_img[y_start:y_start+new_h, x_start:x_start+new_w] = resized
    
    return new_img

# Thêm hàm refine_mask (file: old_model.py)
def refine_mask(mask):
    # Chuyển mask numpy thành ảnh nhị phân 0-255
    binary_mask = (mask * 255).astype(np.uint8)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Làm mờ biên
    refined = cv2.GaussianBlur(refined, (7, 7), 0)
    
    # Chuyển lại thành mask [0, 1]
    return refined.astype(np.float32) / 255.0

# Kiểm tra GPU và tăng tốc độ
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# Load mô hình
model = deeplabv3_resnet50(pretrained=False)
model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)
model.load_state_dict(torch.load("deeplabv3_person.pth", map_location=device))
model.eval().to(device)

# Độ phân giải đích
TARGET_SIZE = (256, 144)  # (width, height)

# Load nền ảo và resize
virtual_bg = cv2.imread("uocj.jpg")
virtual_bg = cv2.resize(virtual_bg, TARGET_SIZE)

# Transform đầu vào
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 144)

# Biến đo FPS
fps_counter = 0
fps = 0
start_time = time.time()


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # frame_resized = cv2.resize(frame, TARGET_SIZE)
    frame = cv2.flip(frame, 1)
    # frame_resized = crop_center_image(frame, target=TARGET_SIZE)
    frame_resized = smart_resize(frame, TARGET_SIZE)
    input_tensor = transform(frame_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out']
        mask = torch.argmax(output.squeeze(), dim=0).byte().cpu().numpy()

    # binary_mask = np.repeat(mask[:, :, None], 3, axis=2)
    refined_mask = refine_mask(mask)
    binary_mask = np.repeat(refined_mask[:, :, None], 3, axis=2)
    result = frame_resized * binary_mask + virtual_bg * (1 - binary_mask)
    result = result.astype(np.uint8)

    # Tính FPS
    fps_counter += 1
    if time.time() - start_time > 1:
        fps = fps_counter
        fps_counter = 0
        start_time = time.time()
    
    # Hiển thị FPS
    cv2.putText(result, f"FPS: {fps}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("DeepLabV3 Virtual Background HD", result)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
