#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : plot_middle_result.py
# Author            : WangZi <wangzitju@163.com>
# Date              : 06.12.2020
# Last Modified Date: 06.12.2020
# Last Modified By  : WangZi <wangzitju@163.com>
"""
展示模型预测结果，取二维检测bbox的结果，
在测试图像中画出, 模型预测的点的坐标和类别
用法：

# 真实测试集
python plot_middle_result.py\
        ./submission/points_logits_real_20201206-1318.json\
        ./data/speed/annos/wz_real_test.json\
        ./data/speed/images/real_test\
        ./data/show_middle_result/real
# 仿真测试集
python plot_middle_result.py\
        ./submission/points_logits_synt_20201206-1318.json\
        ./data/speed/annos/wz_synt_test.json\
        ./data/speed/images/test\
        ./data/show_middle_result/synt
"""
import argparse
import os
import os.path as osp
import cv2
import json
import numpy as np
import pandas
from tqdm import tqdm
from collections import defaultdict
from utils.utils import COLORS_BGR


def parser_args():
    parser = argparse.ArgumentParser('Plot Result')
    parser.add_argument(
        'log_file', type=str,
        default='./submission/points_logits_real_20201206-1318.json',
        help='submission csv file'
    )
    parser.add_argument(
        'bbox_file', type=str,
        default='./data/speed/annos/wz_real_test.json',
        help='bounding box annotation file path'
    )
    parser.add_argument(
        'img_dir', type=str,
        default='./data/speed/images/real_test',
        help='image dir'
    )
    parser.add_argument(
        'out_dir', type=str,
        default='./data/show_middle_result',
        help='output dir for plotted images'
    )
    args = parser.parse_args()
    return args


def read_log_result(file_path):
    with open(file_path, 'r') as f:
        logs = json.load(f)
    return logs
    # for img_name, points_logits in logs.items():
    #     points, logits = [], []
    #     for item in points_logits:
    #         points += item['points']
    #         logits += item['logits']

    #     logs[img_name] = {'points': points, 'logits': logits}
    # return logs


def read_bbox(file_path):
    with open(file_path, 'r') as f:
        bbox_ann = json.load(f)
    bbox_ann = {filename: bbox[0] for filename, bbox in bbox_ann.items()}
    return bbox_ann


def save_img(out_dir, filename, img):
    filename = filename.replace('jpg', 'png')
    save_path = osp.join(out_dir, filename)
    cv2.imwrite(save_path, img)


def ensure_dir(path):
    if not osp.exists(path):
        print("{:s} not exist, make it!".format(path))
        os.makedirs(path, exist_ok=True)


def plot_bbox_and_kpt(img, bbox, multi_points, multi_logits):
    img_show = img.copy()
    # crop bbox
    x1, y1, x2, y2 = [int(item) for item in bbox[:4]]
    obj_pts_original = defaultdict(list)
    for points, logits in zip(multi_points, multi_logits):
        points, logits = np.asarray(points), np.asarray(logits)
        labels, scores = logits.argmax(), logits.max()
        fg_indices = labels != logits.shape[0] - 1
        points, labels, scores =\
            points[fg_indices], labels[fg_indices], scores[fg_indices]

        for pt_, l_ in zip(points, labels):
            obj_pts_original[l_].append(pt_)

    for labels, points in obj_pts_original.items():
        for pt in points:
            x_, y_ = int(pt[0]), int(pt[1])
            cv2.circle(img_show, (x_, y_), 5, COLORS_BGR[labels], 9)
        # plot points
    return img_show


if __name__ == '__main__':
    args = parser_args()
    log_result = read_log_result(args.log_file)
    bbox_result = read_bbox(args.bbox_file)

    ensemble_log = {}
    single_log = {}
    for img_name, points_logits in log_result.items():
        points, logits = [], []
        single_log[img_name] = {
            'points': points_logits[0]['points'],
            'logits': points_logits[0]['logits']
        }
        for item in points_logits:
            points += item['points']
            logits += item['logits']
        ensemble_log[img_name] = {'points': points, 'logits': logits}

    # make save dir for ensemble and single model result
    img_dir, out_dir = args.img_dir, args.out_dir
    sub_dirs = ['ensemble', 'single']
    for _dirs in sub_dirs:
        save_dir = osp.join(out_dir, _dirs)
        ensure_dir(save_dir)

    # 2Dbbox的标注里面只有真实或者仿真图像的标注，
    # 因此，从bbox的标注里开始循环
    for filename, bbox in tqdm(bbox_result.items()):
        img_path = osp.join(img_dir, filename)
        img = cv2.imread(img_path)

        # 1. plot ensemble
        save_path = osp.join(out_dir, 'ensemble', filename)
        points = ensemble_log[filename]['points']
        logits = ensemble_log[filename]['logits']
        img_show = plot_bbox_and_kpt(img, bbox, points, logits)
        cv2.imwrite(save_path, img_show)

        # 2. plot the first model
        save_path = osp.join(out_dir, 'single', filename)
        points = single_log[filename]['points']
        logits = single_log[filename]['logits']
        img_show = plot_bbox_and_kpt(img, bbox, points, logits)
        cv2.imwrite(save_path, img_show)

