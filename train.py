# Huấn luyện DeepLabV3 để phân đoạn người (thay thế Mediapipe)

import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Kiểm tra thiết bị (CPU hoặc GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset tùy chỉnh
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, images, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (144, 256))
        mask = cv2.resize(mask, (144, 256))

        if self.transform:
            image = self.transform(image)

        mask = torch.tensor(mask, dtype=torch.long)
        mask = (mask > 127).long()

        return image, mask

def compute_iou(pred, target, num_classes=2):
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))  # bỏ qua lớp không có
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)
val_mean_ious = []
loss_history = []

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Dataset và DataLoader
BASE_DATA_PATH = 'D:/ZangDev/CodeByZang/Phenikaa/ComputerVision/Project/bigdata/'
IMAGE_DIR = os.path.join(BASE_DATA_PATH, 'images')
MASK_DIR = os.path.join(BASE_DATA_PATH, 'masks')
all_images = os.listdir(IMAGE_DIR)
train_images, test_images = train_test_split(all_images, test_size=0.2, random_state=42)
train_dataset = SegmentationDataset(IMAGE_DIR, MASK_DIR, train_images, transform=transform)
test_dataset = SegmentationDataset(IMAGE_DIR, MASK_DIR, test_images, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Tải mô hình DeepLabV3 pretrained, bỏ aux_classifier nếu không cần thiết
model = deeplabv3_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
model.aux_classifier = None
model = model.to(device)

# Huấn luyện
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    epoch_loss = 0
    for img, mask in tqdm(train_loader):
        img, mask = img.to(device), mask.to(device)
        output = model(img)['out']
        loss = loss_fn(output, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    loss_history.append(epoch_loss / len(train_loader))

    val_iou = []
    model.eval()
    with torch.no_grad():
        for img, mask in test_loader:
            img, mask = img.to(device), mask.to(device)
            output = model(img)['out']
            pred = torch.argmax(output, dim=1)
            iou = compute_iou(pred.cpu(), mask.cpu())
            val_iou.append(iou)
    mean_iou = np.nanmean(val_iou)
    val_mean_ious.append(mean_iou)
    print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(train_loader):.4f} | Val mIoU = {mean_iou:.4f}")

# Lưu mô hình
torch.save(model.state_dict(), "deeplabv3_person.pth")

import matplotlib.pyplot as plt

plt.plot(val_mean_ious, marker='o')
plt.plot(loss_history)
plt.title("Validation Mean IoU per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Mean IoU")
plt.grid(True)
plt.show()
