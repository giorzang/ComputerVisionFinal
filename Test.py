# Thay nền thời gian thực sử dụng mô hình DeepLabV3 với độ phân giải cao 540x960

import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50

def crop_center_image(img, target=(144, 256)):
    h, w = img.shape[:2]
    crop_width, crop_height = target
    
    if w < crop_width or h < crop_height:
        print(f"Ảnh quá nhỏ")
        return

    x_start = (w - crop_width) // 2
    y_start = (h - crop_height) // 2

    cropped_img = img[y_start:y_start+crop_height, x_start:x_start+crop_width]
    return cropped_img

# Load mô hình
model = deeplabv3_resnet50(pretrained=False)
model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)
model.load_state_dict(torch.load("deeplabv3_person.pth", map_location="cpu"))
model.eval()

# Độ phân giải đích
TARGET_SIZE = (144, 144)  # (width, height)

# Load nền ảo và resize
virtual_bg = cv2.imread("uocj.jpg")
virtual_bg = cv2.resize(virtual_bg, TARGET_SIZE)

# Transform đầu vào
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((144, 144)),
    transforms.ToTensor(),
])

# Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 144)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 144)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # frame_resized = cv2.resize(frame, TARGET_SIZE)
    frame_resized = crop_center_image(frame, target=TARGET_SIZE)
    input_tensor = transform(frame_resized).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)['out']
        mask = torch.argmax(output.squeeze(), dim=0).byte().numpy()

    binary_mask = np.repeat(mask[:, :, None], 3, axis=2)
    result = frame_resized * binary_mask + virtual_bg * (1 - binary_mask)
    result = result.astype(np.uint8)

    cv2.imshow("DeepLabV3 Virtual Background HD", result)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
