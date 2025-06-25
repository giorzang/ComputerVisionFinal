import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate, Blur, GaussNoise
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler

from dataset import SegmentationDataset, collate_fn_pad
from loss import CombinedLoss
from utils import compute_iou
from model_init import get_model

# Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transform huấn luyện
train_transform = Compose([
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.5),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.7),
    Blur(blur_limit=3, p=0.3),
    GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    # GaussNoise(mean=0, std=10.0, p=0.3),
    ToTensorV2()
])
test_transform = Compose([ToTensorV2()])

# Dataset
BASE_DATA_PATH = '/media/giorzang/Data-SSD/ZangDev/CodeByZang/Phenikaa/ComputerVision/Project/bigdata/'
IMAGE_DIR = os.path.join(BASE_DATA_PATH, 'images')
MASK_DIR = os.path.join(BASE_DATA_PATH, 'masks')
all_images = os.listdir(IMAGE_DIR)
train_images, test_images = train_test_split(all_images, test_size=0.2, random_state=42)

train_dataset = SegmentationDataset(IMAGE_DIR, MASK_DIR, train_images, transform=train_transform)
test_dataset = SegmentationDataset(IMAGE_DIR, MASK_DIR, test_images, transform=test_transform)
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_pad)
# test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn_pad)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_pad)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn_pad)

# Model, loss, optimizer, scheduler
model = get_model().to(device)
loss_fn = CombinedLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
scaler = GradScaler()

# Training loop
val_mean_ious = []
loss_history = []
best_iou = 0

for epoch in range(20):
    model.train()
    epoch_loss = 0

    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()

        with autocast():
            output = model(img)['out']
            loss = loss_fn(output, mask)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
    
    loss_history.append(epoch_loss / len(train_loader))

    # Validation
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
    
    print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(train_loader):.4f} | Val mIoU = {mean_iou:.4f}")
    scheduler.step(mean_iou)

    if mean_iou > best_iou:
        best_iou = mean_iou
        torch.save(model.state_dict(), "deeplabv3_person_best_mixed.pth")

# Graph
plt.plot(val_mean_ious, marker='o', label='Val mIoU')
plt.plot(loss_history, label='Train Loss')
plt.title("Validation Mean IoU & Train Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
