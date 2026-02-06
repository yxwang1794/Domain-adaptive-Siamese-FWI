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
    ...

class FrequencyAttentionFusion(nn.Module):
    ...


# ------------------- Test -------------------
if __name__ == "__main__":
    B, C, H, W = 1, 1, 200, 2000
    model = SiameseNetwork(D1=5)
    x1 = torch.randn(B, C, H, W)
    x2 = torch.randn(B, C, H, W)
    f1, f2 = model(x1, x2)
    print("Output shape:", f1.shape, f2.shape)



