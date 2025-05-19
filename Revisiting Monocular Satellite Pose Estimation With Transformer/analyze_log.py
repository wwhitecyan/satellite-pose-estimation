#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : analyze_log.py
# Author            : WangZi <wangzitju@163.com>
# Date              : 16.12.2020
# Last Modified Date: 16.12.2020
# Last Modified By  : WangZi <wangzitju@163.com>
"""# noqa: E501
分析log中的train loss和得分
用法：
python analyze_log.py ./work_dirs/split_1/log.txt\
        -item train_loss score
{'train_lr': 9.999999999999845e-06,
 'train_class_error': 97.16625669553831,
 'train_loss': 3.2284546206305333,
 'train_loss_ce': 2.3791822674754144,
 'train_loss_points': 0.8492723514099378,
 'train_loss_ce_unscaled': 2.3791822674754144,
 'train_class_error_unscaled': 97.16625669553831,
 'train_loss_points_unscaled': 0.16985447012313135,
 'train_cardinality_error_unscaled': 7.159860229706979,
 'epoch': 0,
 'n_parameters': 16311566,
 'test_class_error': 99.93215714639692,
 'test_loss': 2.7174181048549824,
 'test_loss_ce': 2.3680067133547653,
 'test_loss_points': 0.34941139750516237,
 'test_loss_ce_unscaled': 2.3680067133547653,
 'test_class_error_unscaled': 99.93215714639692,
 'test_loss_points_unscaled': 0.06988227940095004,
 'test_cardinality_error_unscaled': 10.944776912233722,
 'test_speed_eval_pose': 'tvec score: 1.000000, quat score: 3.141593, final score: 4.141593; median tvec: 1.000000, median quat: 3.141593; mean tvec abs: [0.221831, 0.280029, 10.871020], median tvec abs:[0.159037, 0.189468, 9.843988]'}
"""
import argparse
import re
from matplotlib import pyplot as plt
import json
from matplotlib.colors import TABLEAU_COLORS
from collections import defaultdict
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser('Analyze Log')
    parser.add_argument('log_file', type=str,
                        default='./work_dirs/split_1/log.txt',
                        help='log file to be analyzed')
    parser.add_argument('-item', nargs='+', type=str,
                        default=['train_loss', 'score'])
    parser.add_argument('-save', type=str, default='',
                        help='save name for the plot')
    args = parser.parse_args()
    return args


def read_log(file_path):
    result = []
    with open(file_path, 'r') as f:
        f.readline()  # skip the first row for Namespace
        content = f.readlines()
    content = [json.loads(item.strip()) for item in content]
    content = [item for item in content if 'epoch' in item]
    for item in content:
        eval_pose = item['test_speed_eval_pose']
        number_all = re.findall(r"(?<![a-zA-Z:])[-+]?\d*\.?\d+", eval_pose)
        tmp = {
            'epoch': int(item['epoch']),
            'ce_loss': float(item['train_loss_ce']),
            'point_loss': float(item['train_loss_points']),
            'train_loss': float(item['train_loss']),
            'test_loss': float(item['test_loss']),
            't_score': float(number_all[0]),
            'q_score': float(number_all[1]),
            'score': float(number_all[2])
        }
        result.append(tmp)
    return result


def plot_log(args):
    colors = list(TABLEAU_COLORS.keys())
    raw_log = read_log(args.log_file)
    epoches = np.asarray([raw['epoch'] for raw in raw_log])

    plt.rcParams['font.sans-serif'] = ['Songti SC',
                                       'STFangsong', 'STHeiti', 'BiauKai']
    plt.figure(figsize=(8.5, 6.5), dpi=100)

    content = defaultdict(list)
    legend = []
    for idx, item in enumerate(args.item):
        content[item] = np.asarray([raw[item] for raw in raw_log])
        plt.plot(epoches, content[item], color=colors[idx])
        legend.append(item)
    plt.ylim(0, 4)
    plt.legend(legend, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # # 设置坐标标签字体大小
    ax = plt.gca()
    ax.set_xlabel('epoch', fontsize=12)
    ax.set_ylabel('score', fontsize=12)
    plt.show()
    if args.save != '':
        plt.savefig(args.save)


if __name__ == '__main__':
    args = parse_args()
    plot_log(args)
