#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : plot_result.py
# Author            : WangZi <wangzitju@163.com>
# Date              : 06.12.2020
# Last Modified Date: 06.12.2020
# Last Modified By  : WangZi <wangzitju@163.com>
"""
展示位姿估计结果，在测试图像中画出
1. 目标检测框
2. 关键点的重投影
用法：

# 真实测试集
python plot_result.py\
        ./submission/submission_20201204-1546_w_ensemble.csv\
        ./data/speed/annos/wz_real_test.json\
        ./data/speed/images/real_test\
        ./data/show_result
# 真实测试集
python plot_result.py\
        ./submission/submission_20201204-1546_w_ensemble.csv\
        ./data/speed/annos/wz_synt_test.json\
        ./data/speed/images/test\
        ./data/show_result
"""
import cv2
import json
import os.path as osp
import os
import argparse
import pandas as pd
from tqdm import tqdm

from utils.utils import project_pts_quat_tvec, COLORS_BGR
from utils.speed_eval import PoseSolver


W_pts = PoseSolver().W_Pt


def parser_args():
    parser = argparse.ArgumentParser('Plot Result')
    parser.add_argument(
        'csv_file', type=str,
        default='./submission/submission_20201204-1546_w_ensemble.csv',
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
        default='./data/show_result',
        help='output dir for plotted images'
    )
    args = parser.parse_args()
    return args


def read_csv(file_path):
    result = pd.read_csv(file_path, header=None)
    result = result.to_numpy()
    result = {item[0]: item[1:].tolist() for item in result}
    return result


def read_bbox(file_path):
    with open(file_path, 'r') as f:
        bbox_ann = json.load(f)
    bbox_ann = {filename: bbox[0] for filename, bbox in bbox_ann.items()}
    return bbox_ann


def plot_bbox_and_kpt(img, bbox, quat, tvec):
    img_show = img.copy()

    # plot bbox
    x1, y1, x2, y2 = [int(item) for item in bbox[:4]]
    cv2.rectangle(img_show, (x1, y1), (x2, y2), (0, 255, 0), 4)

    # plot keypoints
    img_pts = project_pts_quat_tvec(W_pts, quat, tvec)
    for pt_idx, point in enumerate(img_pts):
        x1, y1 = [int(item) for item in point]
        cv2.circle(img_show, (x1, y1), 7, COLORS_BGR[pt_idx], 10, cv2.LINE_8)
    return img_show


def save_img(out_dir, filename, img):
    filename = filename.replace('jpg', 'png')
    save_path = osp.join(out_dir, filename)
    cv2.imwrite(save_path, img)


if __name__ == '__main__':
    args = parser_args()
    submission_result = read_csv(args.csv_file)
    print('submission length: {:d}'.format(len(submission_result)))
    bbox_result = read_bbox(args.bbox_file)

    img_dir, out_dir = args.img_dir, args.out_dir
    if not osp.exists(out_dir):
        print("{:s} not exist, make it!".format(out_dir))
        os.makedirs(out_dir, exist_ok=True)

    # 2Dbbox的标注里面只有真实或者仿真图像的标注，
    # 因此，从bbox的标注里开始循环
    for filename, bbox in tqdm(bbox_result.items()):
        img_path = osp.join(img_dir, filename)
        save_path = osp.join(out_dir, filename)

        quat_tvec = submission_result[filename]
        quat, tvec = quat_tvec[:4], quat_tvec[4:]

        img = cv2.imread(img_path)

        img_show = plot_bbox_and_kpt(img, bbox, quat, tvec)
        cv2.imwrite(save_path, img_show)

