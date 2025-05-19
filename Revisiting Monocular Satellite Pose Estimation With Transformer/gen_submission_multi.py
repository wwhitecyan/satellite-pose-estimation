# -*- coding: utf-8 -*-
# File              : gen_submission_multi.py
# Author            : WangZi <wangzitju@163.com>
# Date              : 04.12.2020
# Last Modified Date: 06.12.2020
# Last Modified By  : WangZi <wangzitju@163.com>
"""
根据多个模型生成real test和synt test的结果，用于提交
用法：
python gen_submission_multi.py\
    --input_size 224 --batch_size 20\
    --backbone resnet50s8 --num_queries 30\
    --position_embedding sine\
    --enc_layers 4 --dec_layers 4\
    --resume\
        ./work_dirs/train_ed4_resnet50s8_l2_1/checkpoint.pth\
        ./work_dirs/train_ed4_resnet50s8_l2_2/checkpoint.pth\
        ./work_dirs/train_ed4_resnet50s8_l2_3/checkpoint.pth\
        ./work_dirs/train_ed4_resnet50s8_l2_4/checkpoint.pth\
        ./work_dirs/train_ed4_resnet50s8_l2_5/checkpoint.pth\
        ./work_dirs/train_ed4_resnet50s8_l2_6/checkpoint.pth

"""
import os.path as osp
import json
import cv2
import torch
import random
import numpy as np
import argparse

from utils.submission import SubmissionWriter
from datasets.speed import SpeedSubmission
from models import build_model
import utils.misc as utils
from utils.speed_eval import Multi_Mean_PoseSolver
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader


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

    # ransac
    parser.add_argument('--repro', type=int, default=25,
                        help='reprojectionError')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output_dir', default='./submission',
                        help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='',
                        nargs='+',
                        help='resume from multi checkpoint')
    parser.add_argument('--num_workers', default=2, type=int)

    return parser


@torch.no_grad()
def gen_prediction(model, postprocessors, device, data_loader, prediction):
    """根据当前模型生成预测值
    和用单个模型不太一样，这里仅将模型
    """

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, targets in metric_logger.log_every(
            data_loader, 10, header):
        samples = samples.to(device)
        targets_fileanems = [item.pop('filename') for item in targets]
        targets_clip_bbox = [item.pop('clip_bbox') for item in targets]

        outputs = model(samples)

        results = postprocessors['points'](outputs, targets_clip_bbox)
        for filename, ret in zip(targets_fileanems, results):
            prediction[filename].append(ret)


def gen_submission(prediction, solver):
    """ 根据多个模型的预测值，集成求解位姿
    Input:
        prediction: {
        ''filename': [
                {
                    # 每个模型的预测值
                    'logits': np.ndarray(Nx12),
                    'points': np.ndarray(Nx2)
                },
                ...
                {
                    'logits': np.ndarray(Nx12),
                    'points': np.ndarray(Nx2)
                },
            ]
        }
    Output:
        log:{
        'filename':{
                'quat_pr': [q0, q1, q2, q3],
                'tvec_pr': [v1, v2, v3]
            }
        }
    """
    log = {}
    for filename, pre_list in prediction.items():
        multi_points, multi_logits = [], []
        for item in pre_list:
            multi_logits.append(item['logits'])
            multi_points.append(item['points'])
        try:
            quat_pr, tvec_pr = solver(multi_points, multi_logits)
        except IndexError as e:
            quat_pr, tvec_pr = np.zeros(4), np.zeros(3)
        except cv2.error as e:
            quat_pr, tvec_pr = np.zeros(4), np.zeros(3)
        log[filename] = {
            'quat_pr': np.around(quat_pr, decimals=6).tolist(),
            'tvec_pr': np.around(tvec_pr, decimals=6).tolist(),
        }
    return log


def save_prediction(prediction, save_path):
    log = {}
    for filename, pre_list in prediction.items():
        log[filename] = [
            {
                'points': np.around(item['points'], decimals=6).tolist(),
                'logits': np.around(item['logits'], decimals=6).tolist()
            } for item in pre_list
        ]
    with open(save_path, 'w') as f:
        json.dump(log, f)


def main(args):
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(args.device)
    output_dir = args.output_dir

    # dataset and dataloader
    # 1. annotation files and image dir
    real_test_ann_file = 'wz_real_test.json'
    real_test_img_dir = 'images/real_test'
    synt_test_ann_file = 'wz_synt_test.json'
    synt_test_img_dir = 'images/test'

    # 2. make real dataset and loader
    real_dataset_subm = SpeedSubmission(
        real_test_ann_file, real_test_img_dir, args.input_size)
    real_sampler_val = torch.utils.data.SequentialSampler(real_dataset_subm)
    real_data_loader_subm = DataLoader(
        real_dataset_subm, args.batch_size,
        sampler=real_sampler_val, drop_last=False,
        collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # 3. make synt dataset and loader
    synt_dataset_subm = SpeedSubmission(
        synt_test_ann_file, synt_test_img_dir, args.input_size)
    synt_sampler_val = torch.utils.data.SequentialSampler(synt_dataset_subm)
    synt_data_loader_subm = DataLoader(
        synt_dataset_subm, args.batch_size,
        sampler=synt_sampler_val, drop_last=False,
        collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # container for predcited points and logits
    synt_prediction, real_prediction =\
        defaultdict(list), defaultdict(list)

    solver = Multi_Mean_PoseSolver(args)

    # model, loss, postprocessor and solver
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    n_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # writer
    writer = SubmissionWriter()

    # eval
    print('Start Submission Generating')
    # 1. collect model prediction
    for model_path in args.resume:
        # resume
        print('load model from: [{:s}]'.format(model_path))
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(
            checkpoint['model'], strict=True)
        gen_prediction(model, postprocessors, device,
                       real_data_loader_subm, real_prediction)
        gen_prediction(model, postprocessors, device,
                       synt_data_loader_subm, synt_prediction)

    # save predictions
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    synt_log_save_path = osp.join(output_dir, 'points_logits_synt_{:}.json')
    real_log_save_path = osp.join(output_dir, 'points_logits_real_{:}.json')
    save_prediction(synt_prediction, synt_log_save_path.format(timestamp))
    save_prediction(real_prediction, real_log_save_path.format(timestamp))

    # 2. generate submission
    real_sub = gen_submission(real_prediction, solver)
    synt_sub = gen_submission(synt_prediction, solver)
    for filename, prediction in real_sub.items():
        writer.append_real_test(
            filename,
            prediction['quat_pr'],
            prediction['tvec_pr']
        )
    for filename, prediction in synt_sub.items():
        writer.append_test(
            filename,
            prediction['quat_pr'],
            prediction['tvec_pr']
        )

    # export to csv
    writer.export(out_dir='./submission', suffix=timestamp)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        'DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    assert args.resume != '', 'resume mis in submission!'
    assert isinstance(
        args.resume, list), 'resume from one chpt use gen_single.sh'

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
