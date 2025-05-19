#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : gen_submission_single.py
# Author            : WangZi <wangzitju@163.com>
# Date              : 03.12.2020
# Last Modified Date: 03.12.2020
# Last Modified By  : WangZi <wangzitju@163.com>
"""
根据单个模型生成real test和synt test的结果，用于提交
用法：
python gen_submission_single.py\
    --input_size 224 --batch_size 20\
    --backbone resnet50s8 --num_queries 30\
    --position_embedding sine\
    --enc_layers 4 --dec_layers 4\
    --resume ./work_dirs/train_ed4_resnet50s8/checkpoint.pth
可见 gen_single.sh
"""
import argparse
import numpy as np
import random
import torch
import cv2
from torch.utils.data import DataLoader
from pathlib import Path
import time

from utils.speed_eval import build_solver
import utils.misc as utils
from models import build_model
from datasets.speed import SpeedSubmission
from utils.submission import SubmissionWriter


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
    parser.add_argument('--repro', type=int, default=20,
                        help='reprojectionError')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--num_workers', default=2, type=int)

    return parser


@torch.no_grad()
def gen_submission(model, criterion, postprocessors, ann_file, img_dir,
                   solver, device, args):

    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter(
    #     'class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    metric_logger.add_meter(
        'model_time', utils.SmoothedValue(window_size=20, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'solver_time', utils.SmoothedValue(window_size=20, fmt='{value:.6f}'))

    dataset_subm = SpeedSubmission(
        ann_file, img_dir, args.input_size)

    sampler_val = torch.utils.data.SequentialSampler(dataset_subm)
    data_loader_subm = DataLoader(
        dataset_subm, args.batch_size, sampler=sampler_val, drop_last=False,
        collate_fn=utils.collate_fn, num_workers=args.num_workers)

    log = {}
    for samples, targets in metric_logger.log_every(
            data_loader_subm, 10, header):
        samples = samples.to(device)
        targets_fileanems = [item.pop('filename') for item in targets]
        targets_clip_bbox = [item.pop('clip_bbox') for item in targets]

        end = time.time()
        outputs = model(samples)
        model_time = time.time() - end

        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {
        #     k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if
        #     k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        results = postprocessors['points'](outputs, targets_clip_bbox)
        res = {
            filename: ret for filename, ret in zip(
                targets_fileanems, results
            )
        }

        end = time.time()
        for filename, ret in res.items():
            try:
                quat_pr, tvec_pr = solver(ret['points'], ret['logits'])
            except IndexError as e:
                quat_pr, tvec_pr = np.zeros(4), np.zeros(3)
            except cv2.error as e:
                quat_pr, tvec_pr = np.zeros(4), np.zeros(3)

            log[filename] = {
                'quat_pr': np.around(quat_pr, decimals=6).tolist(),
                'tvec_pr': np.around(tvec_pr, decimals=6).tolist(),
            }
        solver_time = time.time() - end
        metric_logger.update(
            model_time=model_time,
            solver_time=solver_time
        )

    return log


def main(args):

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(args.device)

    # dataset and dataloader
    real_test_ann_file = 'wz_real_test.json'
    real_test_img_dir = 'images/real_test'
    synt_test_ann_file = 'wz_synt_test.json'
    synt_test_img_dir = 'images/test'

    # model, loss, postprocessor and solver
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    n_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad)

    print('number of params:', n_parameters)
    solver = build_solver(args)

    # resume
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(
        checkpoint['model'], strict=True)

    # writer
    writer = SubmissionWriter()

    # eval
    print('Start Submission Generating')

    print('-----------real-------------')
    real_sub = gen_submission(
        model,  criterion, postprocessors,
        real_test_ann_file, real_test_img_dir,
        solver, args.device, args)
    print('-----------synt-------------')
    synt_sub = gen_submission(
        model,  criterion, postprocessors,
        synt_test_ann_file, synt_test_img_dir,
        solver, args.device, args)

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

    writer.export()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        'DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    assert args.resume != '', 'resume mis in submission!'

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
