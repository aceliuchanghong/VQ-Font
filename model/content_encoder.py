import torch.nn as nn
from functools import partial
from model.modules.blocks import ConvBlock


class ContentEncoder(nn.Module):
    """
    nn.Sequential 用于快速构建顺序神经网络。它接受多个 nn.Module 子类的实例作为参数，并将它们按传入顺序依次连接起来
    非常适合用于构建简单的前馈神经网络
    使用 nn.Sequential 能够避免编写冗长的 forward 方法，每一层的前向计算顺序由 nn.Sequential 内部自动处理
    CNN eg:
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(32 * 14 * 14, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    """
    def __init__(self, layers, sigmoid=False):
        super().__init__()
        self.net = nn.Sequential(*layers)
        self.sigmoid = sigmoid

    def forward(self, x):
        out = self.net(x)
        if self.sigmoid:
            out = nn.Sigmoid()(out)
        return out


def content_enc_builder(C_in, C, C_out, norm='none', activ='relu', pad_type='reflect', content_sigmoid=False):
    """
    pad_type 填充类型，reflect表示反射填充
    C_in=1 输入通道数。
    C=32 中间层通道数。
    C_out=256 输出通道数

    partial 是 Python 标准库 functools 模块中的一个函数，用于固定一个函数的一些参数，并返回一个新的函数
    def func(a, b, c):
        return a + b + c
    # 使用 partial 固定参数 b 和 c
    new_func = partial(func, b=2, c=3)
    # 调用 new_func 时，只需传入剩余的参数 a
    result = new_func(1)  # 等同于 func(1, 2, 3)
    print(result)  # 输出 6
    """
    ConvBlk = partial(ConvBlock, norm=norm, activ=activ, pad_type=pad_type)

    layers = [
        ConvBlk(C_in, C, 3, 1, 1, norm='in', activ='relu'),
        ConvBlk(C * 1, C * 2, 3, 2, 1),  # 64x64
        ConvBlk(C * 2, C * 4, 3, 2, 1),  # 32x32
        ConvBlk(C * 4, C * 8, 3, 2, 1),  # 16x16
        ConvBlk(C * 8, C_out, 3, 1, 1)
    ]

    return ContentEncoder(layers, content_sigmoid)
