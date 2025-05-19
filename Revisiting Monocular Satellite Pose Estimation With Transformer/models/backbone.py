# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from functools import partial

from utils.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool,
                 num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and\
               'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {
                "layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            # return_layers = {'layer4': "0"}
            return_layers = {'layer3': "0"}
        self.body = IntermediateLayerGetter(
            backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(
                m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 batchnorm: nn.Module):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=batchnorm)
        # num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        num_channels = 512 if name in ('resnet18', 'resnet34') else 1024
        super().__init__(
            backbone, train_backbone, num_channels, return_interm_layers)


class Backbone8s(nn.Module):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 batchnorm: nn.Module):
        super(Backbone8s, self).__init__()
        backbone = getattr(torchvision.models, 'resnet50')(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=batchnorm)
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and\
               'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        # num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        return_layers = {"layer2": "0", "layer3": "1"}

        self.body = IntermediateLayerGetter(
            backbone, return_layers=return_layers)

        self.up16sto8s = nn.UpsamplingBilinear2d(scale_factor=2)
        self.s8_latern = nn.Conv2d(512, 256, 1, 1, bias=False)
        self.s16_latern = nn.Conv2d(1024, 256, 3, 1, 1, bias=False)
        self.output_conv = nn.Conv2d(512, 512, 3, 1, 1)
        self.num_channels = 512

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        m = tensor_list.mask
        assert m is not None

        out: Dict[str, NestedTensor] = {}
        xs8, xs16 = xs['0'], xs['1']
        xs8 = self.s8_latern(xs8)
        xs16 = self.s16_latern(self.up16sto8s(xs16))
        xsout = self.output_conv(torch.cat([xs8, xs16], 1))

        mask = F.interpolate(
            m[None].float(), size=xsout.shape[-2:]).to(torch.bool)[0]

        out['0'] = NestedTensor(xsout, mask)

        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


class MyGroupNorm(nn.GroupNorm):
    def __init__(self, num_features, num_groups=32):
        super(MyGroupNorm, self).__init__(num_groups, num_features)


def build_batchnorm(args):
    if args.bn == 'frozen_bn':
        return FrozenBatchNorm2d
    elif args.bn == 'sync_bn':
        return nn.SyncBatchNorm
    elif args.bn == 'group_bn':
        return MyGroupNorm
    elif args.bn == 'bn':
        return nn.BatchNorm2d


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    batchnorm = build_batchnorm(args)
    train_backbone = args.lr_backbone > 0
    if args.backbone in ('resnet18', 'resnet34', 'resnet50'):
        backbone = Backbone(
            args.backbone, train_backbone, False, args.dilation, batchnorm)
    else:
        # backbone = Resnet18(256)
        args.backbone = 'resnet50'
        backbone = Backbone8s(
            args.backbone, train_backbone, False, args.dilation, batchnorm)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
