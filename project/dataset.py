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
        image = cv2.resize(image, (540, 960))
        mask = cv2.resize(mask, (540, 960))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image'].float() / 255.0
            mask = augmented['mask'].long()

        mask = (mask > 127).long()
        return image, mask

def pad_to_max_shape(tensor, max_h, max_w):
    _, h, w = tensor.shape
    pad_h = max_h - h
    pad_w = max_w - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom))

def collate_fn_pad(batch):
    # batch = [(img1, mask1), (img2, mask2), ...]
    images, masks = zip(*batch)
    
    # Tìm kích thước lớn nhất trong batch
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded_images = [pad_to_max_shape(img, max_h, max_w) for img in images]
    padded_masks = [pad_to_max_shape(mask.unsqueeze(0).float(), max_h, max_w).squeeze(0).long() for mask in masks]

    return torch.stack(padded_images), torch.stack(padded_masks)
