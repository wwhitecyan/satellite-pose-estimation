#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : plot_enc_num.py
# Author            : WangZi <wangzitju@163.com>
# Date              : 10.12.2020
# Last Modified Date: 10.12.2020
# Last Modified By  : WangZi <wangzitju@163.com>
import argparse
import numpy as np
import os.path as osp
import json
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from pathlib import Path
import re

"""
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


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-output_dir', type=str, default='./show_analyze',
                        help='output_dir for analyze log')
    args = parser.parse_args()
    return args


def read_log(file_path):
    result = []
    with open(file_path, 'r') as f:
        content = f.readlines()
    content = [json.loads(item.strip()) for item in content]
    content = [item for item in content if 'epoch' in item]
    for item in content:
        eval_pose = item['test_speed_eval_pose']
        number_all = re.findall(r"(?<![a-zA-Z:])[-+]?\d*\.?\d+", eval_pose)
        tmp = {
            'epoch': int(item['epoch']),
            'train_loss': float(item['train_loss']),
            'test_loss': float(item['test_loss']),
            'score': float(number_all[2])
        }
        result.append(tmp)
    return result


def plot_pro(file_list, output_dir, name):
    result_all = [read_log(item) for item in file_list]

    colors = list(TABLEAU_COLORS.keys())

    # first plot all
    plt.rcParams['font.sans-serif'] = ['Songti SC',
                                       'STFangsong', 'STHeiti', 'BiauKai']
    plt.figure(figsize=(8.5, 6.5), dpi=500)
    legend = []
    for idx, result in enumerate(result_all):
        epoch = np.asarray([item['epoch'] for item in result])
        score = np.asarray([item['score'] for item in result])
        assert len(epoch) == len(score)
        use_flag = score < 4
        epoch, score = epoch[use_flag], score[use_flag]
        # plt.plot(epoch, score, markers[idx])
        plt.plot(epoch, score, color=colors[idx])
        legend.append('{:s}: {:d}'.format(name, idx+1))
    plt.legend(legend, fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    # # 设置坐标标签字体大小
    ax = plt.gca()
    ax.set_xlabel('epoch', fontsize=22)
    ax.set_ylabel('score', fontsize=22)
    save_name = osp.join(output_dir, name+'_epoch_score_all.png')
    plt.savefig(save_name)
    # plt.show()
    plt.clf()

    # plot the last 20 epoch
    plt.figure(figsize=(5, 3), dpi=500)
    # legend = []
    last_num = 30
    for idx, result in enumerate(result_all):
        epoch = np.asarray([item['epoch'] for item in result])
        score = np.asarray([item['score'] for item in result])
        assert len(epoch) == len(score)
        use_flag = score < 4
        epoch, score = epoch[use_flag][-last_num:], score[use_flag][-last_num:]
        # plt.plot(epoch, score, markers[idx])
        plt.plot(epoch, score, color=colors[idx])
        # legend.append(f'layer: {idx:d}')
    # plt.legend(legend)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    save_name = osp.join(output_dir, name+'_epoch_score_last.png')
    plt.savefig(save_name)
    # plt.show()
    # plt.clf()


def plot_simple(scores, xticks, xlabels, ylabels, name):
    plt.rcParams['font.sans-serif'] = ['Songti SC',
                                       'STFangsong', 'STHeiti', 'BiauKai']
    fig = plt.figure(figsize=(6.5, 5.3), dpi=100)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, right=1,
    #                     left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    plt.bar(x=xticks, height=scores, width=0.4)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # # 设置坐标标签字体大小
    ax = plt.gca()
    ax.set_xlabel("编、解码器层数", fontsize=18)
    ax.set_ylabel('得分', fontsize=18)

    save_name = osp.join(args.output_dir, name+'_num_bar.pdf')
    plt.savefig(save_name)


if __name__ == '__main__':
    args = parser_args()
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. 画bar
    # 1.1 number of layers
    scores = [0.047592, 0.036293, 0.033156, 0.031438, 0.029481, 0.037165]
    xticks = range(1, 7)
    xlabels = "编、解码器层数"
    ylabels = "得分"
    name = 'layer'
    plot_simple(scores, xticks, xlabels, ylabels, name)

    # 1.2 number of query
    scores = [0.033270, 0.032832, 0.032626, 0.032303, 0.032654, 0.031967]
    xticks = (25, 30, 35, 40, 45, 50)
    xlabels = "查询向量个数"
    ylabels = "得分"
    name = 'query'
    plot_simple(scores, xticks, xlabels, ylabels, name)

    # 1.3 input size
    scores = [0.030272, 0.029067, 0.027976, 0.026852, 0.025109, 0.025368]
    xticks = [192, 208, 224, 240, 256, 272]
    xlabels = "输入图像尺寸"
    ylabels = "得分"
    name = 'size'
    plot_simple(scores, xticks, xlabels, ylabels, name)

    # 画曲线图
    # 2.1 number of layers
    layer_file_list = [
        './work_dirs_analyze/num_layers/train_ed1_resnet50s8_l2_1/log.txt',
        './work_dirs_analyze/num_layers/train_ed2_resnet50s8_l2_1/log.txt',
        './work_dirs_analyze/num_layers/train_ed3_resnet50s8_l2_1/log.txt',
        './work_dirs_analyze/num_layers/train_ed4_resnet50s8_l2_1/log.txt',
        './work_dirs_analyze/num_layers/train_ed5_resnet50s8_l2_1/log.txt',
        './work_dirs_analyze/num_layers/train_ed6_resnet50s8_l2_1/log.txt',
    ]
    plot_pro(layer_file_list, output_dir, 'layer')
    # 2.2 number of query
    query_file_list = [
        './work_dirs_analyze/num_query/train_ed4_resnet50s8_25_l2_1/log.txt',
        './work_dirs_analyze/num_query/train_ed4_resnet50s8_30_l2_1/log.txt',
        './work_dirs_analyze/num_query/train_ed4_resnet50s8_35_l2_1/log.txt',
        './work_dirs_analyze/num_query/train_ed4_resnet50s8_40_l2_1/log.txt',
        './work_dirs_analyze/num_query/train_ed4_resnet50s8_45_l2_1/log.txt',
        './work_dirs_analyze/num_query/train_ed4_resnet50s8_50_l2_1/log.txt',
    ]
    plot_pro(query_file_list, output_dir, 'query')
    # 2.3 number of input size
    size_file_list = [
        './work_dirs_analyze/input_size/train_ed4_resnet50s8_query_40_input_192_l2_1/log.txt',
        './work_dirs_analyze/input_size/train_ed4_resnet50s8_query_40_input_208_l2_1/log.txt',
        './work_dirs_analyze/input_size/train_ed4_resnet50s8_query_40_input_224_l2_1/log.txt',
        './work_dirs_analyze/input_size/train_ed4_resnet50s8_query_40_input_240_l2_1/log.txt',
        './work_dirs_analyze/input_size/train_ed4_resnet50s8_query_40_input_256_l2_1/log.txt',
        './work_dirs_analyze/input_size/train_ed4_resnet50s8_query_40_input_272_l2_1/log.txt',
    ]
    plot_pro(size_file_list, output_dir, 'input_size')
