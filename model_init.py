import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

def get_model(num_classes=2, pretrained=True):
    model = deeplabv3_resnet50(pretrained=pretrained)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier = None
    return model
