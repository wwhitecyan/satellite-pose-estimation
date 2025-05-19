# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import utils.misc as utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from utils.speed_eval import build_solver


def map_static_dicts(model, state_dict):
    # 1. update backbone
    # collect state_dict
    backbone_dict = {}
    for name, weight in state_dict.items():
        if name.startswith('backbone'):
            new_name = name.replace('backbone.', '')
            backbone_dict[new_name] = weight

    backbone = model.__dict__['_modules']['backbone']
    msg = backbone.load_state_dict(backbone_dict, False)
    print(msg)

    # 2. transformer
    # collect transformer
    transformer_dict = {}
    for name, weight in state_dict.items():
        if name.startswith('transformer'):
            new_name = name.replace('transformer.', '')
            transformer_dict[new_name] = weight
    transformer = model.__dict__['_modules']['transformer']
    msg = transformer.load_state_dict(transformer_dict, False)
    print(msg)

    # 3.input_proj
    input_proj_dict = {}
    for name, weight in state_dict.items():
        if name.startswith('input_proj'):
            new_name = name.replace('input_proj.', '')
            input_proj_dict[new_name] = weight
    original_proj_num = input_proj_dict['weight'].shape[1]
    input_proj = model.__dict__['_modules']['input_proj']
    new_input_proj_num = input_proj.weight.shape[1]
    assert original_proj_num > new_input_proj_num,\
        'only support new_input_proj_num smaller that 2048'

    if original_proj_num != new_input_proj_num:
        print('original_proj_num: {:d},'
              ' change to new_query_embed_num:{:d}'.format(
                original_proj_num, new_input_proj_num))
        input_proj_dict['weight'] =\
            input_proj_dict['weight'][:, :new_input_proj_num]
    msg = input_proj.load_state_dict(input_proj_dict, False)
    print(msg)

    # 4.query_embed
    query_embed_dict = {}
    for name, weight in state_dict.items():
        if name.startswith('query_embed'):
            new_name = name.replace('query_embed.', '')
            query_embed_dict[new_name] = weight
            original_query_embed_num = weight.shape[0]
    query_embed = model.__dict__['_modules']['query_embed']
    new_query_embed_num = query_embed.weight.shape[0]

    assert original_query_embed_num > new_query_embed_num,\
        'only support query number smaller that 100 now!'

    if original_query_embed_num != new_query_embed_num:
        print('original_query_embed_num: {:d},'
              ' change to new_query_embed_num:{:d}'.format(
                original_query_embed_num, new_query_embed_num))

        query_embed_dict['weight'] =\
            query_embed_dict['weight'][:new_query_embed_num]

    msg = query_embed.load_state_dict(query_embed_dict, False)
    print(msg)


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=[80, 120], type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--input_size', type=int, default=512,
                        help='input image size')

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

    # dataset parameters
    parser.add_argument('--train_ann_file', default='wz_train.json')
    parser.add_argument('--train_index_file', default='train_1.txt')
    parser.add_argument('--train_img_dir', default='images/train')

    parser.add_argument('--val_ann_file', default='wz_train.json')
    parser.add_argument('--val_index_file', default='val_1.txt')
    parser.add_argument('--val_img_dir', default='images/train')
    parser.add_argument('--gt_file', default='train.json')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # ransac
    parser.add_argument('--repro', type=int, default=20,
                        help='reprojectionError')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.output_dir and utils.is_main_process():
        with (Path(args.output_dir) / "log.txt").open("a") as f:
            f.write(str(args) + '\n')

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters()
                    if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, args.lr_drop)

    dataset_train = build_dataset(args=args, train=True)
    dataset_val = build_dataset(args=args, train=False)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(
        dataset_val, args.batch_size, sampler=sampler_val, drop_last=False,
        collate_fn=utils.collate_fn, num_workers=args.num_workers)

    gt_file = args.gt_file
    solver = build_solver(args)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        if "detr-r50-e632da11.pth" in args.resume:
            map_static_dicts(model_without_ddp, checkpoint['model'])
        else:
            model_without_ddp.load_state_dict(
                checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler'\
           in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, speed_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val,
            gt_file, solver, device, args.output_dir)

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
        if args.output_dir and utils.is_main_process():
            with open(output_dir / "eval.json", 'w') as f:
                json.dump(speed_evaluator.log, f)
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            # if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
            if (epoch + 1) % 100 == 0:
                checkpoint_paths.append(
                    output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        # evaluate
        # if epoch > 20:
        test_stats, speed_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val,
            gt_file, solver, device, args.output_dir
        )
        for k, v in test_stats.items():
            log_stats.update({f'test_{k}': v})

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # TODO evaluator
            # for evaluation logs
            # if epoch > 20:
            if speed_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                filenames = ['latest.json']
                if epoch % 50 == 0:
                    filenames.append(f'{epoch:03}.json')
                for name in filenames:
                    with open(output_dir / 'eval' / name, 'w') as f:
                        json.dump(speed_evaluator.log, f)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
