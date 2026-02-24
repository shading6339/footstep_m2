"""
models.py
---------
比較実験用モデル群。
  - SimpleCNN  : 卒研ベースライン（同構造）
  - ResNet18   : torchvision pretrained
  - EnvNetV2   : 波形直接入力 (未実装・プレースホルダー)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


# ---------- SimpleCNN（卒研ベースライン）----------

class SimpleCNN(nn.Module):
    """
    卒研と同一構造の 3 層 CNN。
    入力: (B, 3, 256, 256)
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ---------- ResNet18（転移学習）----------

class ResNet18Classifier(nn.Module):
    """
    ImageNet pretrained ResNet18 の全結合層だけ差し替え。
    入力: (B, 3, 256, 256)
    """

    def __init__(self, num_classes: int, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# ---------- ファクトリ ----------

def build_model(name: str, num_classes: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "simplecnn":
        return SimpleCNN(num_classes)
    elif name == "resnet18":
        return ResNet18Classifier(num_classes, **kwargs)
    else:
        raise ValueError(f"未知のモデル名: {name}。SimpleCNN / ResNet18 から選択してください。")
