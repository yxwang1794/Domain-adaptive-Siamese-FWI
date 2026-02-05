import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainDiscriminator2D(nn.Module):
    """输入: [B,1,H,W]  输出: [B,2] (0=obs, 1=pred)"""
    def __init__(self, in_ch=1, base=32, p=0.1):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=(5,15), stride=(2,4), padding=(2,7), bias=False),
            nn.BatchNorm2d(base), nn.GELU(),
            nn.Conv2d(base, base*2, kernel_size=(3,11), stride=(2,4), padding=(1,5), bias=False),
            nn.BatchNorm2d(base*2), nn.GELU(),
            nn.Conv2d(base*2, base*4, kernel_size=(3,7), stride=(2,2), padding=(1,3), bias=False),
            nn.BatchNorm2d(base*4), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # -> [B, base*4, 1, 1]
        )
        self.head = nn.Sequential(
            nn.Flatten(),                          # [B, base*4]
            nn.Dropout(p),
            nn.Linear(base*4, 128),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(128, 2)                      # obs/pred
        )

    def forward(self, x):
        return self.head(self.feat(x))  # [B,2]