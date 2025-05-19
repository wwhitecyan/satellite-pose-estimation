#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : test_dataset.py
# Author            : WangZi <wangzitju@163.com>
# Date              : 21.11.2020
# Last Modified Date: 21.11.2020
# Last Modified By  : WangZi <wangzitju@163.com>
from datasets.speed import SpeedTrain, make_transforms, SpeedSubmission
from tqdm import tqdm
import numpy as np
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser('Test Speed Dataset')
    parser.add_argument('-type', type=str, default='train',
                        choices=('train', 'submission'),
                        help='test dataset type')
    args = parser.parse_args()
    return args


def test_train():
    ann_file = 'wz_train.json'
    # index_file = 'val_1.txt'
    index_file = 'train_1.txt'
    img_dir = 'images/train'
    train_flag = True
    transforms = make_transforms(train_flag)
    dataset = SpeedTrain(ann_file, index_file, img_dir, train_flag, transforms)
    print(len(dataset))

    cv2.namedWindow('win')
    cv2.moveWindow('win', 10, 10)
    key, wait = 0, 0

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    for idx in tqdm(range(len(dataset))):
        # target = dataset[idx]
        # cropImg, landmarks = target['image'], target['keypoints']

        cropImg, target = dataset[idx]
        landmarks = target['landmarks']
        for i_, m_, s_ in zip(cropImg, mean, std):
            i_.mul_(s_).add_(m_).mul_(255)
        cropImg = cropImg.permute(1, 2, 0)
        cropImg = np.asarray(cropImg).astype(np.uint8)
        landmarks = np.asarray(landmarks) * cropImg.shape[0]
        for pt in landmarks:
            cv2.circle(cropImg, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), 5)
        cv2.imshow('win', cropImg)
        key = cv2.waitKey(wait)
        if key == 27:
            break
        if key == 13:
            wait = 33
        if key == ord(' '):
            wait = 0
    cv2.destroyWindow('win')


def test_submission():
    real_test_ann_file = 'wz_real_test.json'
    real_test_img_dir = 'images/real_test'

    dataset = SpeedSubmission(real_test_ann_file, real_test_img_dir)
    print(len(dataset))

    cv2.namedWindow('win')
    cv2.moveWindow('win', 10, 10)
    key, wait = 0, 0

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    for idx in tqdm(range(len(dataset))):

        cropImg, target = dataset[idx]
        for i_, m_, s_ in zip(cropImg, mean, std):
            i_.mul_(s_).add_(m_).mul_(255)
        cropImg = cropImg.permute(1, 2, 0)
        cropImg = np.asarray(cropImg).astype(np.uint8)
        cv2.imshow('win', cropImg)
        key = cv2.waitKey(wait)
        if key == 27:
            break
        if key == 13:
            wait = 33
        if key == ord(' '):
            wait = 0
    cv2.destroyWindow('win')


if __name__ == '__main__':
    args = parse_args()
    if args.type == 'train':
        test_train()
    else:
        test_submission()
