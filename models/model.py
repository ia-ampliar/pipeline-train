import torch
import torch.nn as nn
from torchvision import models

# MobileNetV2 com pesos pré-treinados
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(MobileNetV2, self).__init__()
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.last_channel, num_classes)

    def forward(self, x):
        return self.mobilenet(x)

# ResNet50 com pesos pré-treinados
class ResNet50(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# VGG16 com pesos pré-treinados
class VGG16(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(VGG16, self).__init__()
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.vgg(x)

# AlexNet com pesos pré-treinados
class AlexNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(AlexNet, self).__init__()
        self.alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
        self.alexnet.classifier[6] = nn.Linear(self.alexnet.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.alexnet(x)

# EfficientNet com pesos pré-treinados
class EfficientNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNet, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)

# DenseNet com pesos pré-treinados
class DenseNet121(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(DenseNet121, self).__init__()
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)

    def forward(self, x):
        return self.densenet(x)