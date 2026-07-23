import torch
import torch.nn as nn


class DynamicCWRUModel(nn.Module):
    """动态 CNN-1D 多任务模型。

    参数：
        params: dict，由 search_config.suggest_params() 生成的超参数字典
        input_length: 输入波形长度（默认 2048）
        num_type_classes: 故障类型类别数（默认 4）
        num_size_classes: 故障尺寸类别数（默认 4）
    """

    def __init__(self, params, input_length=2048, num_type_classes=4, num_size_classes=4):
        super().__init__()
        self.params = params
        self.input_length = input_length

        # ---- 构建编码器 ----
        encoder_layers = []
        in_channels = 1  # 单通道波形输入
        num_layers = params['num_conv_layers']
        channels = params['channels']

        for i in range(num_layers):
            out_channels = channels[i]

            if i == 0:
                # 第1层：宽卷积核 + 大步长，无 padding（valid 卷积）
                conv = nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=params['first_kernel'],
                    stride=params['first_stride'],
                    padding=0
                )
            else:
                # 后续层：标准卷积核 + stride=1 + same padding
                ks = params['kernel_size']
                conv = nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=ks,
                    stride=1,
                    padding=ks // 2
                )

            encoder_layers.append(conv)
            encoder_layers.append(nn.BatchNorm1d(out_channels))
            encoder_layers.append(nn.ReLU(inplace=True))

            # 可选池化层
            if params.get('use_pool', False):
                encoder_layers.append(nn.MaxPool1d(
                    kernel_size=params['pool_size'],
                    stride=params['pool_stride']
                ))

            in_channels = out_channels

        # 全局平均池化 + 展平
        encoder_layers.append(nn.AdaptiveAvgPool1d(1))
        encoder_layers.append(nn.Flatten())

        self.encoder = nn.Sequential(*encoder_layers)

        # 计算编码器输出维度
        self.encoder_output_dim = self._compute_output_dim()

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

        # 权重初始化
        self._init_weights()

    def _compute_output_dim(self):
        """通过前向传播 dummy 张量计算编码器输出维度。

        这比手动计算更可靠，因为能准确处理各种 padding/stride/pool 组合。
        """
        with torch.no_grad():
            dummy = torch.zeros(2, 1, self.input_length)
            output = self.encoder(dummy)
            return output.shape[1]

    def _init_weights(self):
        """Kaiming 初始化卷积层和线性层，BatchNorm 初始化为 weight=1, bias=0"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """前向传播。

        输入：
            x: (batch, 1, input_length) 波形张量
        输出：
            type_logits: (batch, num_type_classes) 故障类型 logits
            size_logits: (batch, num_size_classes) 故障尺寸 logits
        """
        features = self.encoder(x)
        type_logits = self.type_head(features)
        size_logits = self.size_head(features)
        return type_logits, size_logits


def build_model(params, input_length=2048, num_type_classes=4, num_size_classes=4):
    """便捷函数：根据超参数字典构建模型。

    参数：
        params: dict，超参数字典
        input_length: 输入波形长度
        num_type_classes: 故障类型类别数
        num_size_classes: 故障尺寸类别数

    返回：
        DynamicCWRUModel 实例
    """
    return DynamicCWRUModel(params, input_length, num_type_classes, num_size_classes)


if __name__ == '__main__':
    pass