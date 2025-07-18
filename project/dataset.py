import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

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
        # Không cần resize vì đã chuẩn hóa toàn bộ dataset thành độ phân giải 540x960
        # image = cv2.resize(image, (540, 960))
        # mask = cv2.resize(mask, (540, 960))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image'].float() / 255.0
            mask = augmented['mask'].long()

        mask = (mask > 127).long()
        return image, mask