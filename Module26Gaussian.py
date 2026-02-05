import numpy as np
import random
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, D1):
        super(SiameseNetwork, self).__init__()
        self.D1 = D1
        self.channel = 5

        # ===== 主干 CNN 层 =====
        self.cnn1 = nn.Conv2d(self.channel, self.D1, kernel_size=3, stride=1, padding=1)
        self.a1 = nn.LeakyReLU(0.5)

        self.cnn2 = nn.Conv2d(self.D1, 2 * self.D1, kernel_size=3, stride=1, padding=1)
        self.a2 = nn.LeakyReLU(0.5)

        self.cnn3 = nn.Conv2d(2 * self.D1, 2 * self.D1, kernel_size=3, stride=1, padding=1)
        self.a3 = nn.LeakyReLU(0.5)

        self.cnn4 = nn.Conv2d(2 * self.D1, 4 * self.D1, kernel_size=3, stride=1, padding=1)
        self.a4 = nn.LeakyReLU(0.5)

        self.cnn5 = nn.Conv2d(4 * self.D1, 4 * self.D1, kernel_size=3, stride=1, padding=1)
        self.a5 = nn.LeakyReLU(0.5)

        self.cnn6 = nn.Conv2d(4 * self.D1, 2 * self.D1, kernel_size=3, stride=1, padding=1)
        self.a6 = nn.LeakyReLU(0.5)

        self.cnn7 = nn.Conv2d(2 * self.D1, self.D1, kernel_size=3, stride=1, padding=1)
        self.a7 = nn.LeakyReLU(0.5)

        self.cnn8 = nn.Conv2d(self.D1, 1, kernel_size=3, stride=1, padding=1)

        # ===== 残差卷积 (xx1~xx7 + xx) =====
        self.cnnXX1 = nn.Conv2d(self.channel, self.D1, kernel_size=3, stride=1, padding=1)
        self.axx1 = nn.LeakyReLU(0.5)

        self.cnnXX2 = nn.Conv2d(self.channel, 2 * self.D1, kernel_size=3, stride=1, padding=1)
        self.axx2 = nn.LeakyReLU(0.5)

        self.cnnXX3 = nn.Conv2d(self.channel, 2 * self.D1, kernel_size=3, stride=1, padding=1)
        self.axx3 = nn.LeakyReLU(0.5)

        self.cnnXX4 = nn.Conv2d(self.channel, 4 * self.D1, kernel_size=3, stride=1, padding=1)
        self.axx4 = nn.LeakyReLU(0.5)

        self.cnnXX5 = nn.Conv2d(self.channel, 4 * self.D1, kernel_size=3, stride=1, padding=1)
        self.axx5 = nn.LeakyReLU(0.5)

        self.cnnXX6 = nn.Conv2d(self.channel, 2 * self.D1, kernel_size=3, stride=1, padding=1)
        self.axx6 = nn.LeakyReLU(0.5)

        self.cnnXX7 = nn.Conv2d(self.channel, self.D1, kernel_size=3, stride=1, padding=1)
        self.axx7 = nn.LeakyReLU(0.5)

        self.cnnXX = nn.Conv2d(self.channel, self.D1, kernel_size=3, stride=1, padding=1)
        self.axx = nn.LeakyReLU(0.5)

        # ===== 新增：后处理投影层 =====
        self.post_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.5, inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.5, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            #nn.AdaptiveAvgPool2d((8, 8)),   # 压缩特征图，避免输入直接泄漏
            #nn.Flatten(),
            #nn.Linear(16 * 8 * 8, 128),     # 投影到 embedding 空间
            #nn.LeakyReLU(0.5, inplace=True)
        )


    def forward(self, x):
        # ===== 残差分支 =====
        xx1 = self.axx1(self.cnnXX1(x))
        xx2 = self.axx2(self.cnnXX2(x))
        xx3 = self.axx3(self.cnnXX3(x))
        xx4 = self.axx4(self.cnnXX4(x))
        xx5 = self.axx5(self.cnnXX5(x))
        xx6 = self.axx6(self.cnnXX6(x))
        xx7 = self.axx7(self.cnnXX7(x))
        xx = self.axx(self.cnnXX(x))

        # ===== 主干 + 残差 =====
        out1 = self.a1(self.cnn1(x)) + xx1
        out2 = self.a2(self.cnn2(out1)) + xx2
        out3 = self.a3(self.cnn3(out2)) + xx3
        out4 = self.a4(self.cnn4(out3)) + xx4
        out5 = self.a5(self.cnn5(out4)) + xx5
        out6 = self.a6(self.cnn6(out5)) + xx6
        out7 = self.a7(self.cnn7(out6)) + xx7
        out8 = self.cnn8(out7)
        out8 = out8 + xx + x  # 保持原始残差结构

        # ===== 新增：后处理（可学习） =====
        #embedding = self.post_layers(out8)   # [B, 128] 向量
        return out8

    # def forward(self, x1, x2):
    #     f1 = self.forward_once(x1)
    #     f2 = self.forward_once(x2)
    #     return f1, f2

class FrequencyAttentionFusion(nn.Module):
    def __init__(self, in_channels=5):
        super(FrequencyAttentionFusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局特征
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, in_channels, bias=False),
            nn.Softmax(dim=1)  # 输出三个频段的权重
        )

        self.feature_extractor0 = SiameseNetwork(D1=5)
        self.feature_extractor1 = SiameseNetwork(D1=5)
        self.feature_extractor2 = SiameseNetwork(D1=5)
        self.feature_extractorf = SiameseNetwork(D1=5)

    def forward_once(self, x):
        x_low = x[2]
        x_mid = x[1]
        x_high = x[0]

        x_mid = F.interpolate(x_mid, size=(400, 2000), mode='bilinear', align_corners=False)
        x_low = F.interpolate(x_low, size=(400, 2000), mode='bilinear', align_corners=False)

        x_low = self.feature_extractor0(x_low)
        x_mid = self.feature_extractor1(x_mid)
        x_high = self.feature_extractor2(x_high)


        # #
        # #
        # #
        x_cat = torch.stack([x_low, x_mid, x_high], dim=1)  # [B,3,H,W]
        w = self.avg_pool(x_cat).squeeze(-1).squeeze(-1)    # [B,3]
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)          # [B,3,1,1]

        x_fused = (x_cat * w).sum(dim=1)                    # 加权融合
        #print(w[0,0,2,0,0],w[0,1,0,0,0],w[0,2,0,0,0])
        #x_fused = 0.1*x_low + 0.2*x_mid + 0.7*x_high    #0.5 1 3
        #x_fused = self.feature_extractorf(x_fused)
        #x_fused += x_fused2
        return x_fused

    def forward(self, x1, x2):

        o1 = self.forward_once(x1)
        o2 = self.forward_once(x2)

        return o1, o2


# ------------------- Test -------------------
if __name__ == "__main__":
    B, C, H, W = 1, 1, 200, 2000
    model = SiameseNetwork(D1=5)
    x1 = torch.randn(B, C, H, W)
    x2 = torch.randn(B, C, H, W)
    f1, f2 = model(x1, x2)
    print("Output shape:", f1.shape, f2.shape)



