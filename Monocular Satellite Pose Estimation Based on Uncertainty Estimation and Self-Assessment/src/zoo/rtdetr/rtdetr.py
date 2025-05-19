"""by lyuwenyu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np


from src.core import register


__all__ = [
    "RTDETR",
]


@register
class RTDETR(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        self.temper_param = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        # x.shape 50,3,256,56
        x = self.backbone(x)
        # x[0].shape 50,128,32,32
        # x[1].shape 50,256,16,16
        # x[2].shape 50,512,8,8
        x = self.encoder(x)
        # x[0].shape 50,256,32,32
        # x[1].shape 50,256,16,16
        # x[2].shape 50,256,8,8
        x = self.decoder(x, targets)

        return x

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
