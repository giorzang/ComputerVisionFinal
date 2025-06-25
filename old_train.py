# Huấn luyện DeepLabV3 để phân đoạn người (thay thế Mediapipe)

import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from sklearn.model_selection import train_test_split
from albumentations import (Compose, HorizontalFlip, RandomBrightnessContrast,
                            ShiftScaleRotate, Blur, GaussNoise)
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Kiểm tra thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
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
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image'].float() / 255.0
            mask = augmented['mask'].long()

        mask = (mask > 127).long()

        return image, mask

# Loss
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, smooth=1e-5):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return 0.7 * self.ce(pred, target) + 0.3 * self.dice(pred, target)

# Metric
def compute_iou(pred, target, num_classes=2):
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

# Augmentation
train_transform = Compose([
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.5),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.7),
    Blur(blur_limit=3, p=0.3),
    # GaussNoise(mean=0, std=10.0, p=0.3), #  
    GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    ToTensorV2()
])

# Dataset và DataLoader
BASE_DATA_PATH = '/media/giorzang/Data-SSD/ZangDev/CodeByZang/Phenikaa/ComputerVision/Project/bigdata/'
IMAGE_DIR = os.path.join(BASE_DATA_PATH, 'images')
MASK_DIR = os.path.join(BASE_DATA_PATH, 'masks')
all_images = os.listdir(IMAGE_DIR)
train_images, test_images = train_test_split(all_images, test_size=0.2, random_state=42)

train_dataset = SegmentationDataset(IMAGE_DIR, MASK_DIR, train_images, transform=train_transform)
test_dataset = SegmentationDataset(IMAGE_DIR, MASK_DIR, test_images, transform=Compose([ToTensorV2()]))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Model
model = deeplabv3_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
model.aux_classifier = None
model = model.to(device)

# Huấn luyện
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

val_mean_ious = []
loss_history = []
best_iou = 0

for epoch in range(20):
    model.train()
    epoch_loss = 0
    for img, mask in tqdm(train_loader):
        img, mask = img.to(device), mask.to(device)
        output = model(img)['out']
        loss = loss_fn(output, mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    loss_history.append(epoch_loss / len(train_loader))

    # Đánh giá
    model.eval()
    val_iou = []
    with torch.no_grad():
        for img, mask in test_loader:
            img, mask = img.to(device), mask.to(device)
            output = model(img)['out']
            pred = torch.argmax(output, dim=1)
            iou = compute_iou(pred.cpu(), mask.cpu())
            val_iou.append(iou)
    mean_iou = np.nanmean(val_iou)
    val_mean_ious.append(mean_iou)

    # In kết quả
    print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(train_loader):.4f} | Val mIoU = {mean_iou:.4f}")

    # Lưu mô hình tốt nhất
    if mean_iou > best_iou:
        best_iou = mean_iou
        torch.save(model.state_dict(), "deeplabv3_person_best.pth")

# Biểu đồ
plt.plot(val_mean_ious, marker='o', label='Val mIoU')
plt.plot(loss_history, label='Train Loss')
plt.title("Validation Mean IoU & Train Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()