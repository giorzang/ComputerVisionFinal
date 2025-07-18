import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

def get_model(num_classes=2, pretrained=True):
    if pretrained:
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
    else:
        weights = None
    model = deeplabv3_resnet50(weights=weights)

    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier = None
    return model
