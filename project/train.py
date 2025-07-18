import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from albumentations import (Compose, HorizontalFlip, RandomBrightnessContrast,
                            ShiftScaleRotate, Blur, GaussNoise)
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from dataset import SegmentationDataset
from loss import CombinedLoss
from utils import compute_iou
from model_init import get_model

# Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Augmentations
train_transform = Compose([
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.5),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.7),
    Blur(blur_limit=3, p=0.3),
    GaussNoise(var_limit=(10.0, 50.0), p=0.3),
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

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

# Model, loss, optimizer, scheduler
model = get_model(pretrained=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = CombinedLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
scaler = torch.amp.GradScaler()

val_mean_ious = []
loss_history = []
best_iou = 0
accumulation_steps = 32

# Training loop
for epoch in range(35):
    model.train()
    epoch_loss = 0
    
    for i, (img, mask) in enumerate(tqdm(train_loader, desc="Training")):
        img, mask = img.to(device), mask.to(device)

        with torch.amp.autocast(device_type=device.type):
            output = model(img)['out']
            loss = loss_fn(output, mask)
            loss /= accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):    
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss.item()
        
    loss_history.append(epoch_loss / len(train_loader))

    # Validation
    model.eval()
    val_iou = []
    with torch.no_grad():
        for img, mask in tqdm(test_loader, desc="Validating"):
            img, mask = img.to(device), mask.to(device)

            with torch.amp.autocast(device_type=device.type):
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
        torch.save(model.state_dict(), "deeplabv3_person_best.pth")
    
    torch.cuda.empty_cache()
    
    # Giải nhiệt CPU/GPU
    time.sleep(60)
    
# Biểu đồ
plt.subplot(121), plt.plot(loss_history, label='Train Loss'), plt.title("Train Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Value")

plt.subplot(122), plt.plot(val_mean_ious, label='Val mIoU'), plt.title("Validation Mean IoU per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Value")

plt.legend()
plt.grid(True)
plt.show()
