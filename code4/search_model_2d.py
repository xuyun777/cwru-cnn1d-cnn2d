import torch
import torch.nn as nn
from search_config_2d import FREQ_BINS, INPUT_LENGTH, N_FFT, HOP_LENGTH

N_FRAMES = (INPUT_LENGTH + N_FFT)/HOP_LENGTH + 1       # 时间帧数（center=True）

def _conv_output_size(size, kernel, stride, padding):
    """计算 Conv2d 输出尺寸（PyTorch 默认公式）"""
    return (size + 2 * padding - kernel) // stride + 1


def _pool_output_size(size, kernel, stride=None):
    """计算 MaxPool2d 输出尺寸"""
    if stride is None:
        stride = kernel
    return size // stride


def _simulate_spatial_dims(in_h, in_w, params):
    """
    模拟 encoder 各层后的空间尺寸 (H, W)。
    
    返回列表，每项为 (H, W)。
    用于判断哪些层可以安全地添加池化。
    """
    h, w = in_h, in_w
    dims = []
    num_layers = params['num_conv_layers']

    for i in range(num_layers):
        if i == 0:
            ks = params['first_kernel']
            stride = params['first_stride']
            padding = ks // 2
        else:
            ks = params['kernel_size']
            stride = 1
            padding = ks // 2

        h = _conv_output_size(h, ks, stride, padding)
        w = _conv_output_size(w, ks, stride, padding)

        if params.get('use_pool', False):
            ps = params['pool_size']
            h_after = _pool_output_size(h, ps)
            w_after = _pool_output_size(w, ps)
            # 仅当池化后两个维度都 ≥ 2 时才记录池化后的尺寸
            if h_after >= 2 and w_after >= 2:
                h, w = h_after, w_after
            # 否则跳过池化，保持当前尺寸

        dims.append((h, w))

    return dims


class DynamicCWRUModel2D(nn.Module):
    """
    Dynamic CNN-2D 多任务模型，用于 CWRU 轴承故障诊断（v3 修复版）。
    
    输入：语谱图 (B, 1, FREQ_BINS, N_FRAMES)，支持任意非方形输入。
    输出：type_logits + size_logits。
    """

    def __init__(self, params, num_type_classes=4, num_size_classes=4,
                 in_h=FREQ_BINS, in_w=N_FRAMES):
        super().__init__()
        self.params = params

        # ---- 模拟各层空间尺寸，决定哪些层加池化 ----
        spatial_dims = _simulate_spatial_dims(in_h, in_w, params)

        # ---- 构建 CNN-2D 编码器 ----
        encoder_layers = []
        in_channels = 1
        num_layers = params['num_conv_layers']
        channels = params['channels']

        h, w = in_h, in_w
        for i in range(num_layers):
            out_channels = channels[i]

            if i == 0:
                ks = params['first_kernel']
                stride = params['first_stride']
                padding = ks // 2
            else:
                ks = params['kernel_size']
                stride = 1
                padding = ks // 2

            encoder_layers.append(nn.Conv2d(in_channels, out_channels,
                                            kernel_size=ks, stride=stride,
                                            padding=padding))
            encoder_layers.append(nn.BatchNorm2d(out_channels))
            encoder_layers.append(nn.ReLU(inplace=True))

            # 更新尺寸
            h = _conv_output_size(h, ks, stride, padding)
            w = _conv_output_size(w, ks, stride, padding)

            if params.get('use_pool', False):
                ps = params['pool_size']
                h_after = _pool_output_size(h, ps)
                w_after = _pool_output_size(w, ps)
                if h_after >= 2 and w_after >= 2:
                    encoder_layers.append(nn.MaxPool2d(kernel_size=ps, stride=ps))
                    h, w = h_after, w_after

            in_channels = out_channels

        # 全局平均池化 → 无论输入尺寸如何，输出 (B, C, 1, 1)
        encoder_layers.append(nn.AdaptiveAvgPool2d(1))
        encoder_layers.append(nn.Flatten())

        self.encoder = nn.Sequential(*encoder_layers)

        # encoder 输出维度 = 最后一个卷积层的通道数
        self.encoder_output_dim = channels[-1]

        # ---- 构建分类头 ----
        self.type_head = nn.Sequential(
            nn.Linear(self.encoder_output_dim, params['type_hidden']),
            nn.ReLU(inplace=True),
            nn.Dropout(params['dropout']),
            nn.Linear(params['type_hidden'], num_type_classes),
        )

        self.size_head = nn.Sequential(
            nn.Linear(self.encoder_output_dim, params['size_hidden']),
            nn.ReLU(inplace=True),
            nn.Dropout(params['dropout']),
            nn.Linear(params['size_hidden'], num_size_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.encoder(x)
        type_logits = self.type_head(features)
        size_logits = self.size_head(features)
        return type_logits, size_logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(params, num_type_classes=4, num_size_classes=4,
                in_h=FREQ_BINS, in_w=N_FRAMES):
    """工厂函数，创建 DynamicCWRUModel2D 实例"""
    return DynamicCWRUModel2D(params,
                              num_type_classes=num_type_classes,
                              num_size_classes=num_size_classes,
                              in_h=in_h, in_w=in_w)


if __name__ == '__main__':
    pass