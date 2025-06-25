import cv2
import torch
import numpy as np
import time
from torchvision import transforms
from model_init import get_model

# ==== Cấu hình ====
MODEL_PATH = "deeplabv3_person_best.pth"
VIRTUAL_BG_PATH = "uocj.jpg"
TARGET_SIZE = (256, 144)

# ==== Thiết bị ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Thiết bị sử dụng: {device}")

# ==== Hàm tiền xử lý ====
def smart_resize(img, target_size=(256, 144)):
    h, w = img.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    new_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_start = (target_w - new_w) // 2
    y_start = (target_h - new_h) // 2
    new_img[y_start:y_start+new_h, x_start:x_start+new_w] = resized
    return new_img

def refine_mask(mask):
    binary_mask = (mask * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)
    refined = cv2.GaussianBlur(refined, (7, 7), 0)
    return refined.astype(np.float32) / 255.0

# ==== Load mô hình ====
model = get_model().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ==== Load nền ảo ====
virtual_bg = cv2.imread(VIRTUAL_BG_PATH)
virtual_bg = cv2.resize(virtual_bg, TARGET_SIZE)

# ==== Transform ảnh đầu vào ====
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# ==== Khởi động webcam ====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_SIZE[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_SIZE[1])

fps_counter = 0
fps = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_resized = smart_resize(frame, TARGET_SIZE)
    input_tensor = transform(frame_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out']
        mask = torch.argmax(output.squeeze(), dim=0).byte().cpu().numpy()

    refined_mask = refine_mask(mask)
    binary_mask = np.repeat(refined_mask[:, :, None], 3, axis=2)
    result = frame_resized * binary_mask + virtual_bg * (1 - binary_mask)
    result = result.astype(np.uint8)

    fps_counter += 1
    if time.time() - start_time > 1:
        fps = fps_counter
        fps_counter = 0
        start_time = time.time()

    cv2.putText(result, f"FPS: {fps}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Virtual Background - DeepLabV3", result)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
