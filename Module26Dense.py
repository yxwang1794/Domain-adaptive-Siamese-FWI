import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union

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


# -------------------- Strong Dense Backbone (2 blocks + transition + gated skip) --------------------
class SimpleDenseBackboneStrong(nn.Module):
    ...

# -------------------- Per-channel band attention fusion --------------------
class FrequencyAttentionFusionBandWise(nn.Module):
    ...

# ------------------- quick test -------------------
if __name__ == "__main__":
    B, C, H, W = 1, 5, 400, 2000

    model = FrequencyAttentionFusionBandWise(
        backbone=SimpleDenseBackboneStrong(
            in_ch=5, base=8, growth=8, L=6, window_k=4,
            out_ch=8, use_norm=True, neg_slope=0.1,
            bottleneck=4, dropout2d=0.05, use_se=True
        ),
        out_ch=8,
        share_backbone=True,
        verbose_shapes=True
    )

    x1 = [torch.randn(B, C, H, W), torch.randn(B, C, H, W), torch.randn(B, C, H, W)]
    x2 = [torch.randn(B, C, H, W), torch.randn(B, C, H, W), torch.randn(B, C, H, W)]
    f1, f2 = model(x1, x2)
    print("Output:", f1.shape, f2.shape, " gate mean:", model.backbone.gate.mean().item())

