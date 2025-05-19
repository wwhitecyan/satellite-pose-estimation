#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : visualize_features.py
# Author            : WangZi <wangzitju@163.com>
# Date              : 09.12.2020
# Last Modified Date: 09.12.2020
# Last Modified By  : WangZi <wangzitju@163.com>
"""
可视化网络学习到的特征
python visualize_features.py\
    --input_size 224\
    --backbone resnet50s8\
    --num_queries 30\
    --enc_layers 4 --dec_layers 4\
    --resume ./work_dirs/train_ed4_resnet50s8/checkpoint.pth\
    --img_path ./data/speed/images/real_test/img000111real.jpg\
    --bbox_path ./data/speed/annos/wz_real_test.json
"""
import argparse
import torch
import os.path as osp
from PIL import Image
import cv2
import numpy as np
import json
import random
from pathlib import Path
import utils.misc as utils
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt


from models import build_model
from utils.utils import COLORS_BGR


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Visualize Features', add_help=False)
    parser.add_argument('--input_size', type=int, default=224,
                        help='input image size')
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
    parser.add_argument('--output_dir', default='./vis_feat',
                        help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--img_path', type=str,
                        help='image path')
    parser.add_argument('--bbox_path', type=str,
                        help='bbox annotations for image')

    return parser


def read_bbox(file_path):
    with open(file_path, 'r') as f:
        bbox_ann = json.load(f)
    bbox_ann = {filename: bbox[0] for filename, bbox in bbox_ann.items()}
    return bbox_ann


def generate_clip_bbox(bbox, image_size):
    x1, y1, x2, y2 = bbox
    bbox_width, bbox_height = x2 - x1, y2 - y1
    scale = max(bbox_width, bbox_height) * 1.2

    x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
    half_scale = scale / 2

    x1, x2 = x_center - half_scale, x_center + half_scale
    y1, y2 = y_center - half_scale, y_center + half_scale
    bbox_clip = np.asarray([x1, y1, x2, y2])

    bbox_clip[0::2] = bbox_clip[0::2].clip(min=0, max=image_size[0])
    bbox_clip[1::2] = bbox_clip[1::2].clip(min=0, max=image_size[1])
    return bbox_clip


def normalize(img):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    img = F.normalize(img, mean=mean, std=std)
    return img


def prepare_img(args):
    """
    读取图像，并根据bbox进行裁剪
    return:
        1. 原图像
        2. 裁剪后的图像
        3. 归一化的Tensor
    """
    img_path, bbox_path = args.img_path, args.bbox_path

    # 1. read image
    img_name = osp.basename(img_path)
    img = cv2.imread(img_path)
    assert img is not None, '{:s} not exits!'.format(img_name)
    height, width = img.shape[:2]
    # 2. read bbox
    bbox_result = read_bbox(bbox_path)
    assert img_name in bbox_result, '{:s} not in {:s}'.format(
        img_name, bbox_path)
    bbox = bbox_result[img_name]

    # 3.clip image
    resize = (args.input_size, args.input_size)
    bbox_clip = generate_clip_bbox(bbox[:4], (height, width))
    img = Image.fromarray(img)
    img_crop = img.crop(bbox_clip).resize(resize)
    img_crop_tensor = torch.from_numpy(
        np.asarray(img_crop)).to(torch.float).permute(2, 0, 1) / 255
    img_crop_tensor = normalize(img_crop_tensor)
    return np.asarray(img), img_crop, img_crop_tensor


def visualize_dec_features(weight, size, labels, save_dir='./vis_feat'):
    """可视化解码器自注意力模块权重
    weight: (num_queries, size*size)
    """
    assert len(weight) == len(labels)
    save_dir = osp.join(save_dir, 'dec')
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for idx, (w_, l_) in enumerate(zip(weight, labels)):
        save_name = osp.join(
            save_dir, '{:02d}_label_{:02d}.png'.format(idx, int(l_)))
        feat = w_.view(size, size).detach().to('cpu').numpy()
        plt.figure(figsize=(16, 16), dpi=100)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1,
                            left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.imshow(feat)
        plt.axis('off')
        plt.savefig(save_name, dpi=100, edgecolor='b', pad_inches=0)
        plt.clf()

    # fig, axes = plt.subplots(nrows=5, ncols=6)
    # for idx in range(len(weight)):
    #     feat = weight[idx].view(size, size).to('cpu').detach().numpy()
    #     axes[idx // 6, idx % 6].imshow(feat)
    #     axes[idx // 6, idx % 6].axis('off')
    # plt.show()
    # plt.savefig('dec_features.png')


def visualize_enc_features(
        weight, size, points, labels, save_dir='./vis_feat'):
    """可视化编码器自注意力模块权重
    weight: (size*size, size*size)
    size: feature map size, input_size // num_stride
    """
    assert len(points) == len(labels)
    save_dir = osp.join(save_dir, 'enc')
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    weight = weight.reshape(size, size, size, size).detach().to('cpu')
    weight = weight.numpy()

    for idx, (pt_, l_) in enumerate(zip(points, labels)):
        save_name = osp.join(
            save_dir, '{:02d}_label_{:02d}.png'.format(idx, int(l_)))
        plt.figure(figsize=(16, 16), dpi=100)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1,
                            left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.imshow(weight[..., int(pt_[1]), int(pt_[0])])
        plt.axis('off')
        plt.savefig(save_name, dpi=100, edgecolor='b', pad_inches=0)
        plt.clf()


def visualize_image(img, points, labels, save_dir='./vis_feat'):
    """可视化网络输出
        points: num_queries * 2
        labels: num_queries,
    """
    assert len(points) == len(labels)
    # 画出所有
    save_img = osp.join(save_dir, 'img_all_result.png')

    if isinstance(img, Image.Image):
        img_show = np.asarray(img, dtype=np.uint8).copy()
        img_show = img_show[:, :, ::-1]
    img_show = np.asarray(img, dtype=np.uint8).copy()

    for pt_, l_ in zip(points, labels):
        l_ = int(l_)
        x_, y_ = int(pt_[0]), int(pt_[1])
        cv2.circle(img_show, (x_, y_), 2, COLORS_BGR[l_], 2)

    cv2.imwrite(save_img, img_show)

    # 只画出前景
    save_img = osp.join(save_dir, 'img_fg_result.png')
    if isinstance(img, Image.Image):
        img_show = np.asarray(img, dtype=np.uint8).copy()
        img_show = img_show[:, :, ::-1]
    img_show = np.asarray(img, dtype=np.uint8).copy()

    for pt_, l_ in zip(points, labels):
        if l_ == 11:
            continue
        l_ = int(l_)
        x_, y_ = int(pt_[0]), int(pt_[1])
        cv2.circle(img_show, (x_, y_), 2, COLORS_BGR[l_], 2)

    cv2.imwrite(save_img, img_show)


def main(args):
    out_dir = args.output_dir

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build and load models
    device = torch.device(args.device)
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    # resume
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(
        checkpoint['model'], strict=True)

    # prepare image
    img_original, img_crop, img_tensor = prepare_img(args)
    sample = utils.nested_tensor_from_tensor_list([img_tensor])
    # img_crop.show()
    # img_crop.save('img.png')

    # hook the enc and dec features
    dec_attn_weights = []
    enc_attn_weights = []
    model.transformer.decoder.layers[-2].multihead_attn.register_forward_hook(
        lambda self, input, output: dec_attn_weights.append(output[1])
    )
    model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
        lambda self, input, output: enc_attn_weights.append(output[1])
    )

    # forward data
    heat_map_size = args.input_size // 8
    outputs = model(sample.to(args.device))
    # 只有一张图片，取第一个元素
    out_logits = outputs['pred_logits'][0]
    out_points = outputs['pred_points'][0]
    out_logits, out_points = out_logits.to('cpu'), out_points.to('cpu')
    out_points *= heat_map_size

    prob = torch.softmax(out_logits, -1)
    labels = prob.argmax(1)
    # fg_indices = labels != out_logits.shape[1] - 1
    # out_points = out_points[fg_indices]

    dec_attn_weights = dec_attn_weights[0][0]
    enc_attn_weights = enc_attn_weights[0][0]
    visualize_dec_features(dec_attn_weights, heat_map_size, labels, out_dir)
    visualize_enc_features(
        enc_attn_weights, heat_map_size, out_points, labels, out_dir)

    out_points *= 8
    visualize_image(img_crop, out_points, labels, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    assert args.resume != '', 'resume mis in submission!'

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
