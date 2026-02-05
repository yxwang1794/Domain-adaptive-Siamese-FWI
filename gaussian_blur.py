import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianBlur(nn.Module):
    """高斯模糊层，使用固定权重的高斯核"""

    def __init__(self, kernel_size=5, sigma=1.0, channels=5):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels
        self.padding = kernel_size // 2

        # 创建高斯核
        kernel = self.create_gaussian_kernel()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)  # [out_c, in_c, H, W]
        kernel = kernel.repeat(channels, 1, 1, 1)  # 复制到每个输入通道

        # 注册为不可训练参数
        self.register_buffer('weight', kernel)

    def create_gaussian_kernel(self):
        """创建高斯卷积核"""
        # 生成高斯分布的坐标网格
        x = torch.arange(self.kernel_size).float() - self.kernel_size // 2
        y = torch.arange(self.kernel_size).float() - self.kernel_size // 2
        x, y = torch.meshgrid(x, y, indexing='ij')

        # 计算高斯分布
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * self.sigma ** 2))
        kernel = kernel / kernel.sum()  # 归一化
        return kernel

    def forward(self, x):
        """应用高斯模糊"""
        # 深度可分离卷积实现高斯模糊
        return F.conv2d(x,
                        weight=self.weight,
                        padding=self.padding,
                        groups=self.channels)


class LowFrequencyExtractor(nn.Module):
    """低频信息提取网络（简化版）"""

    def __init__(self, pyramid_levels=3):
        super(LowFrequencyExtractor, self).__init__()
        self.pyramid_levels = pyramid_levels
        self.gaussian_blur = GaussianBlur(kernel_size=5, sigma=1.0)

    def extract_low_frequency(self, x):
        """直接提取多尺度低频信息"""
        low_freq_pyramid = []
        current = x

        # 添加原始分辨率下的低频信息
        low_freq_pyramid.append(self.gaussian_blur(current))

        # 逐级下采样并提取低频信息
        for _ in range(self.pyramid_levels):
            # 下采样
            downsampled = F.interpolate(current,
                                        scale_factor=0.5,
                                        mode='bilinear',
                                        align_corners=False)
            #downsampled = current

            # 高斯模糊提取当前尺度的低频信息
            low_freq = self.gaussian_blur(downsampled)
            low_freq_pyramid.append(low_freq)

            # 更新当前图像为下采样后的图像
            current = downsampled

        return low_freq_pyramid

    def forward(self, x):
        """单图像低频信息提取"""
        return self.extract_low_frequency(x)