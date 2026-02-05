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

# -------------------- Norm --------------------
# 你已有的 make_norm / backbone 可以不动，这里只给 Fusion 部分
def make_norm(c: int, use_norm: bool = False, groups: int = 32) -> nn.Module:
    if not use_norm:
        return nn.Identity()
    g = min(groups, c)
    while g > 1 and (c % g) != 0:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=c)

# -------------------- SE (optional) --------------------
class SEBlock(nn.Module):
    def __init__(self, c: int, r: int = 8, neg_slope: float = 0.5):
        super().__init__()
        mid = max(1, c // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, mid, kernel_size=1, bias=True),
            nn.LeakyReLU(neg_slope, inplace=True),
            nn.Conv2d(mid, c, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))

# -------------------- DenseNet-BC core --------------------
class DenseCoreBC(nn.Module):
    """
    Norm -> Act -> 1x1 (bottleneck) -> Norm -> Act -> 3x3 (growth)
    """
    def __init__(
        self,
        in_ch: int,
        growth: int,
        bottleneck: int = 4,          # mid = bottleneck * growth
        use_norm: bool = True,
        neg_slope: float = 0.5,
        dilation: int = 1,
        dropout2d: float = 0.0,
        use_se: bool = True,
        se_r: int = 8,
    ):
        super().__init__()
        mid = bottleneck * growth

        self.norm1 = make_norm(in_ch, use_norm=use_norm)
        self.act1 = nn.LeakyReLU(neg_slope, inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid, kernel_size=1, bias=True)

        self.norm2 = make_norm(mid, use_norm=use_norm)
        self.act2 = nn.LeakyReLU(neg_slope, inplace=True)
        self.conv2 = nn.Conv2d(
            mid, growth,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=True
        )

        self.drop = nn.Dropout2d(p=dropout2d) if dropout2d > 0 else nn.Identity()
        self.se = SEBlock(growth, r=se_r, neg_slope=neg_slope) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(self.act1(self.norm1(x)))
        y = self.conv2(self.act2(self.norm2(x)))
        y = self.drop(y)
        y = self.se(y)
        return y

# -------------------- Window Dense Block (V2) --------------------
class WindowDenseBlockV2(nn.Module):
    """
    改进点：
      - window 模式下永远拼一个 stem_summary（不再丢 stem）
      - block 末尾做 LFF(1x1) 压到 fuse_ch，并加 block residual（RDB风格）
    """
    def __init__(
        self,
        in_ch: int,
        L: int,
        growth: int,
        window_k: Optional[int] = 4,
        fuse_ch: int = 64,
        stem_summary_ch: Optional[int] = None,  # 默认=growth
        bottleneck: int = 4,
        use_norm: bool = True,
        neg_slope: float = 0.5,
        dropout2d: float = 0.0,
        use_se: bool = True,
        se_r: int = 8,
        dilation_cycle: Tuple[int, ...] = (1, 1, 2, 1),
    ):
        super().__init__()
        self.L = int(L)
        self.growth = int(growth)
        self.window_k = window_k

        self.stem_summary_ch = int(stem_summary_ch) if stem_summary_ch is not None else int(growth)
        self.stem_proj = nn.Conv2d(in_ch, self.stem_summary_ch, kernel_size=1, bias=True)

        # 预计算每层 core 输入通道
        in_ch_list = []
        for i in range(self.L):
            if window_k is None:
                # full dense: [stem(in_ch)] + i * growth
                core_in = in_ch + i * growth
            else:
                # window dense: [stem_summary] + min(i, window_k) * growth
                core_in = self.stem_summary_ch + min(i, int(window_k)) * growth
            in_ch_list.append(core_in)

        self.layers = nn.ModuleList()
        for i, core_in in enumerate(in_ch_list):
            dil = dilation_cycle[i % len(dilation_cycle)] if dilation_cycle is not None else 1
            self.layers.append(
                DenseCoreBC(
                    in_ch=core_in,
                    growth=self.growth,
                    bottleneck=bottleneck,
                    use_norm=use_norm,
                    neg_slope=neg_slope,
                    dilation=dil,
                    dropout2d=dropout2d,
                    use_se=use_se,
                    se_r=se_r,
                )
            )

        # raw out ch = last x_in ch + growth
        self.raw_out_ch = in_ch_list[-1] + self.growth

        # Local Feature Fusion + block residual
        self.lff = nn.Conv2d(self.raw_out_ch, fuse_ch, kernel_size=1, bias=True)
        self.res_proj = nn.Conv2d(in_ch, fuse_ch, kernel_size=1, bias=True)

        self.out_ch = fuse_ch

    def forward(self, stem_feat: torch.Tensor) -> torch.Tensor:
        stem_sum = self.stem_proj(stem_feat)
        ys: List[torch.Tensor] = []
        x_out: Optional[torch.Tensor] = None

        for i, core in enumerate(self.layers):
            if self.window_k is None:
                # full dense: concat([stem] + ys)
                x_in = stem_feat if len(ys) == 0 else torch.cat([stem_feat] + ys, dim=1)
            else:
                # window dense: concat([stem_sum] + last ys)
                if len(ys) == 0:
                    x_in = stem_sum
                else:
                    recent = ys[-int(self.window_k):]
                    x_in = torch.cat([stem_sum] + recent, dim=1)

            y = core(x_in)
            ys.append(y)
            x_out = torch.cat([x_in, y], dim=1)

        fused = self.lff(x_out)
        fused = fused + self.res_proj(stem_feat)
        return fused

# -------------------- Strong Dense Backbone (2 blocks + transition + gated skip) --------------------
class SimpleDenseBackboneStrong(nn.Module):
    """
    Input : B×5×H×W
    Output: B×out_ch×H×W
    """
    def __init__(
        self,
        in_ch: int = 5,
        base: int = 32,
        growth: int = 16,
        L: int = 8,                     # 总层数，内部拆成两段
        window_k: Optional[int] = 4,
        out_ch: int = 64,
        use_norm: bool = True,
        neg_slope: float = 0.5,
        bottleneck: int = 4,
        dropout2d: float = 0.05,
        use_se: bool = True,
    ):
        super().__init__()
        L1 = L // 2
        L2 = L - L1

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=3, padding=1, bias=True),
            make_norm(base, use_norm=use_norm),
            nn.LeakyReLU(neg_slope, inplace=True),
        )

        mid_ch = max(out_ch, 4 * base)  # 第一块先拉宽再压回

        self.block1 = WindowDenseBlockV2(
            in_ch=base, L=L1, growth=growth, window_k=window_k,
            fuse_ch=mid_ch, bottleneck=bottleneck,
            use_norm=use_norm, neg_slope=neg_slope,
            dropout2d=dropout2d, use_se=use_se
        )

        self.trans = nn.Sequential(
            nn.Conv2d(mid_ch, base * 2, kernel_size=1, bias=True),
            make_norm(base * 2, use_norm=use_norm),
            nn.LeakyReLU(neg_slope, inplace=True),
        )

        self.block2 = WindowDenseBlockV2(
            in_ch=base * 2, L=L2, growth=growth, window_k=window_k,
            fuse_ch=out_ch, bottleneck=bottleneck,
            use_norm=use_norm, neg_slope=neg_slope,
            dropout2d=dropout2d, use_se=use_se,
            dilation_cycle=(1, 2, 2, 1)  # 第二块给点更大感受野（仍然是DenseNet风格）
        )

        # gated skip：避免捷径覆盖 Dense 表达
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        self.gate = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.block2(self.trans(self.block1(self.stem(x))))
        return feat + self.gate * self.skip(x)

# -------------------- Per-channel band attention fusion --------------------
class FrequencyAttentionFusionBandWise(nn.Module):
    """
    输入: x = [high, mid, low]，每个 B×5×H×W
    输出: B×out_ch×H×W
    注意力：每个频带一个标量（B×3）
    """
    def __init__(
        self,
        backbone: nn.Module,
        in_ch: int = 5,
        out_ch: int = 64,
        attn_hidden: int = 16,
        share_backbone: bool = True,
        use_norm_refine: bool = True,
        neg_slope: float = 0.5,
        verbose_shapes: bool = False,
        attn_mode: str = "meanabs",   # "mean" / "meanabs" / "energy"
        temperature: float = 1.0,     # softmax 温度，<1 更尖锐，>1 更平滑
    ):
        super().__init__()
        self.verbose_shapes = verbose_shapes
        self.out_ch = out_ch
        self.share_backbone = share_backbone
        self.attn_mode = attn_mode
        self.temperature = float(temperature)

        # 频段轻量 adapter（可留可不留；一般留着更稳）
        self.band_adapter = nn.ModuleList([
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True) for _ in range(3)
        ])

        if share_backbone:
            self.backbone = backbone
        else:
            # 不共享的话你需要自己传 3 个 backbone；这里给最小可用写法
            self.backbone0 = backbone
            self.backbone1 = type(backbone)()  # 不建议这样；最好显式传入
            self.backbone2 = type(backbone)()

        # band-wise attention MLP：输入 [B,3] -> 输出 [B,3] -> softmax
        self.fc = nn.Sequential(
            nn.Linear(3, attn_hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(attn_hidden, 3, bias=True),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
            make_norm(out_ch, use_norm=use_norm_refine),
            nn.LeakyReLU(neg_slope, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
        )

    def _extract(self, x_low, x_mid, x_high):
        if self.share_backbone:
            f_low  = self.backbone(x_low)
            f_mid  = self.backbone(x_mid)
            f_high = self.backbone(x_high)
        else:
            f_low  = self.backbone0(x_low)
            f_mid  = self.backbone1(x_mid)
            f_high = self.backbone2(x_high)
        return f_low, f_mid, f_high

    def _band_descriptor(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        x_cat: [B,3,C,H,W] -> s: [B,3]
        """
        if self.attn_mode == "mean":
            # 每个频带整体均值
            return x_cat.mean(dim=(2,3,4))
        elif self.attn_mode == "energy":
            # 能量（更适合振幅变化明显的数据）
            return (x_cat * x_cat).mean(dim=(2,3,4))
        else:
            # 默认 meanabs（通常比 mean 稳）
            return x_cat.abs().mean(dim=(2,3,4))

    def forward_once(self, x: Union[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if not (isinstance(x, (list, tuple)) and len(x) == 3):
            raise TypeError("Expected x as list/tuple of 3 tensors: [high, mid, low]")

        x_high, x_mid, x_low = x[0], x[1], x[2]

        # 对齐尺寸到 high
        target_hw = x_high.shape[-2:]
        if x_mid.shape[-2:] != target_hw:
            x_mid = F.interpolate(x_mid, size=target_hw, mode="bilinear", align_corners=False)
        if x_low.shape[-2:] != target_hw:
            x_low = F.interpolate(x_low, size=target_hw, mode="bilinear", align_corners=False)

        # band adapters（顺序：low, mid, high）
        x_low  = self.band_adapter[0](x_low)
        x_mid  = self.band_adapter[1](x_mid)
        x_high = self.band_adapter[2](x_high)

        f_low, f_mid, f_high = self._extract(x_low, x_mid, x_high)

        if self.verbose_shapes:
            print("f_low:", f_low.shape, "f_mid:", f_mid.shape, "f_high:", f_high.shape)

        # stack 顺序固定为 [low, mid, high]
        x_cat = torch.stack([f_low, f_mid, f_high], dim=1)  # [B,3,C,H,W]

        # band-wise descriptor: [B,3]
        s = self._band_descriptor(x_cat)

        # MLP + temperature softmax -> [B,3]
        logits = self.fc(s) / self.temperature
        w = torch.softmax(logits, dim=1).view(-1, 3, 1, 1, 1)  # [B,3,1,1,1]

        fused = (x_cat * w).sum(dim=1)  # [B,C,H,W]
        fused = self.refine(fused)
        return fused

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

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
