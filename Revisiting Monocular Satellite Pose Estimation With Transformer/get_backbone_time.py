#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : get_backbone_time.py
# Author            : WangZi <wangzitju@163.com>
# Date              : 13.12.2020
# Last Modified Date: 13.12.2020
# Last Modified By  : WangZi <wangzitju@163.com>
"""
测试设计的backbone与原backbone的计算耗时
"""
import time
import argparse
import torch
import numpy as np
import random
from models import build_model
import utils.misc as utils


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Point Set transformer Submission Gen', add_help=False)
    parser.add_argument('--input_size', type=int, default=512,
                        help='input image size')
    parser.add_argument('--batch_size', default=30, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model."
                        " If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in"
                        " the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine',
                        type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use"
                        " on top of the image features")
    parser.add_argument('--bn', type=str, default='frozen_bn',
                        choices=('frozen_bn', 'group_bn', 'sync_bn', 'bn'),
                        help='batch norm type in backbone')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward"
                        " layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings "
                        "(dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside"
                        "the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding"
                        " losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_pts', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--pts_loss_coef', default=5., type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification"
                        "weight of the no-object class")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--test_num', default=100, type=int,
                        help='average image number')

    return parser


def main(args):
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build and load models
    device = torch.device(args.device)
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    input_size = args.input_size

    imgs = torch.rand((3, input_size, input_size), dtype=torch.float)
    sample = utils.nested_tensor_from_tensor_list([imgs])
    sample = sample.to(device)

    model.backbone(sample)
    end = time.time()
    for _ in range(args.test_num):
        model.backbone(sample)
    backbont_time = time.time() - end
    backbont_time /= args.test_num
    print(f'time: {backbont_time:.6f}')


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
