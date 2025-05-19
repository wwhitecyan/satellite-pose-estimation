"""MobileNetV3 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init

from src.core import register

__all__ = ["MobileNetV3_Small", "MobileNetV3_Large"]


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size = max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    """expand + depthwise + pointwise"""

    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        self.conv2 = nn.Conv2d(
            expand_size,
            expand_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=expand_size,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)
        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size),
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_size,
                    out_channels=in_size,
                    kernel_size=3,
                    groups=in_size,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size),
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_size,
                    out_channels=out_size,
                    kernel_size=3,
                    groups=in_size,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        skip = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))

        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)


@register
class MobileNetV3_Large(nn.Module):
    def __init__(
        self,
        depth,
        pretrained,
        variant="d",
        num_stages=4,
        return_idx=[0, 1, 2, 3],
        freeze_at=-1,
        freeze_norm=True,
        act=nn.Hardswish,
    ):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.Bn1 = nn.BatchNorm2d(128)
        self.Bn2 = nn.BatchNorm2d(256)
        self.hs1 = act(inplace=True)
        self.Hs1 = act(inplace=True)
        self.Hs2 = act(inplace=True)

        self.pretrained = pretrained
        self.Conv1 = nn.Conv2d(16, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.Conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU, False, 1),
            Block(3, 16, 64, 24, nn.ReLU, False, 2),
            Block(3, 24, 72, 24, nn.ReLU, False, 1),
            Block(5, 24, 72, 40, nn.ReLU, True, 2),
            Block(5, 40, 120, 40, nn.ReLU, True, 1),
            Block(5, 40, 120, 40, nn.ReLU, True, 1),
            Block(3, 40, 240, 80, act, False, 2),
            Block(3, 80, 200, 80, act, False, 1),
            Block(3, 80, 184, 80, act, False, 1),
            Block(3, 80, 184, 80, act, False, 1),
            Block(3, 80, 480, 112, act, True, 1),
            Block(3, 112, 672, 112, act, True, 1),
            Block(5, 112, 672, 160, act, True, 2),
            Block(5, 160, 672, 160, act, True, 1),
            Block(5, 160, 960, 160, act, True, 1),
        )

        self.conv2 = nn.Conv2d(160, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.hs2 = act(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.linear3 = nn.Linear(960, 1280, bias=False)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = act(inplace=True)
        self.drop = nn.Dropout(0.2)

        # self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        if isinstance(self.pretrained, str):
            params = torch.load(self.pretrained)
            own_state_dict = self.state_dict()
            for name, param in params.items():
                if (
                    name in own_state_dict
                    and param.size() == own_state_dict[name].size()
                ):
                    own_state_dict[name].copy_(param)
                else:
                    print(f"Skipping loading param '{name}': size mismatch.")

            self.load_state_dict(own_state_dict)

    def forward(self, x):
        # out = self.hs1(self.bn1(self.conv1(x)))
        # out = self.bneck(out)

        # out = self.hs2(self.bn2(self.conv2(out)))
        # out = self.gap(out).flatten(1)
        # out = self.drop(self.hs3(self.bn3(self.linear3(out))))

        # CBA x.shape 50,3,256,256
        out = self.hs1(self.bn1(self.conv1(x)))
        # out.shape 50,16,128,128
        b = F.interpolate(out, size=(64, 64), mode="bilinear", align_corners=False)
        b = self.Hs1(self.Bn1(self.Conv1(b)))
        c = self.Hs2(self.Bn2(self.Conv2(b)))
        out = self.bneck(out)
        # out.shape 50,96,8,8

        out = self.hs2(self.bn2(self.conv2(out)))

        return [b, c, out]


@register
class MobileNetV3_Small(nn.Module):
    def __init__(
        self,
        depth,
        pretrained,
        variant="d",
        num_stages=4,
        return_idx=[0, 1, 2, 3],
        freeze_at=-1,
        freeze_norm=True,
        act=nn.Hardswish,
    ):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.Conv1 = nn.Conv2d(16, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.Conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = act(inplace=True)

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU, True, 2),
            Block(3, 16, 72, 24, nn.ReLU, False, 2),
            Block(3, 24, 88, 24, nn.ReLU, False, 1),
            Block(5, 24, 96, 40, act, True, 2),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 120, 48, act, True, 1),
            Block(5, 48, 144, 48, act, True, 1),
            Block(5, 48, 288, 96, act, True, 2),
            Block(5, 96, 576, 96, act, True, 1),
            Block(5, 96, 576, 96, act, True, 1),
        )

        self.conv2 = nn.Conv2d(96, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.hs2 = act(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.linear3 = nn.Linear(576, 1280, bias=False)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = act(inplace=True)
        self.drop = nn.Dropout(0.2)
        self.pretrained = pretrained
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        if isinstance(self.pretrained, str):
            params = torch.load(self.pretrained)
            own_state_dict = self.state_dict()
            for name, param in params.items():
                if (
                    name in own_state_dict
                    and param.size() == own_state_dict[name].size()
                ):
                    own_state_dict[name].copy_(param)
                else:
                    print(f"Skipping loading param '{name}': size mismatch.")

            self.load_state_dict(own_state_dict)

    def forward(self, x):
        # CBA x.shape 50,3,256,256
        out = self.hs1(self.bn1(self.conv1(x)))
        # out.shape 50,16,128,128
        b = F.interpolate(out, size=(64, 64), mode="bilinear", align_corners=False)
        b = self.Conv1(b)
        c = self.Conv2(b)
        out = self.bneck(out)
        # out.shape 50,96,8,8

        out = self.hs2(self.bn2(self.conv2(out)))
        # out = self.gap(out).flatten(1)
        # out = self.drop(self.hs3(self.bn3(self.linear3(out))))

        # Upsample to 50,256,16,16

        # Upsample to 50,128,32,32

        # return self.linear4(out)
        return [b, c, out]
